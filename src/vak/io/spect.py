"""functions for dealing with vocalization datasets as pandas DataFrames"""
from glob import glob
import logging
import os
from pathlib import Path

import dask.bag as db
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd

from .annotation import source_annot_map
from ..config import validators
from ..util.general import timebin_dur_from_vec
from ..util.path import find_audio_fname, array_dict_from_path

# constant, used for names of columns in DataFrame below
DF_COLUMNS = [
    'audio_path',
    'spect_path',
    'annot_path',
    'annot_format',
    'duration',
    'timebin_dur',
]


def to_dataframe(spect_format,
                 spect_dir=None,
                 spect_files=None,
                 annot_list=None,
                 annot_format=None,
                 spect_annot_map=None,
                 labelset=None,
                 n_decimals_trunc=3,
                 freqbins_key='f',
                 timebins_key='t',
                 spect_key='s',
                 audio_path_key='audio_path'
                 ):
    """convert spectrogram files into a dataset of vocalizations represented as a Pandas DataFrame.
    Spectrogram files are array in .npz files created by numpy or in .mat files created by Matlab.

    Parameters
    ----------
    spect_format : str
        format of files containing spectrograms. One of {'mat', 'npz'}
    spect_dir : str
        path to directory of files containing spectrograms as arrays.
        Default is None.
    spect_files : list
        List of paths to array files. Default is None.
    annot_list : list
        of annotations for array files. Default is None
    annot_format : str
        name of annotation format. Added as a column to the DataFrame if specified.
        Used by other functions that open annotation files via their paths from the DataFrame.
        Should be a format that the crowsetta library recognizes.
        Default is None.
    spect_annot_map : dict
        Where keys are paths to files and value corresponding to each key is
        the annotation for that file.
        Default is None.
    labelset : list
        of str or int, set of unique labels for vocalizations. Default is None.
        If not None, skip files where the associated annotations contain labels not in labelset.
    n_decimals_trunc : int
        number of decimal places to keep when truncating the timebin duration calculated from
        the vector of time bins.
        Default is 3, i.e. assumes milliseconds is the last significant digit.
    freqbins_key : str
        key for accessing vector of frequency bins in files. Default is 'f'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    audio_path_key : str
        key for accessing path to source audio file for spectogram in files.
        Default is 'audio_path'.

    Returns
    -------
    vak_df : pandas.Dataframe
        that represents a dataset of vocalizations.

    Notes
    -----
    Each file should contain a spectrogram as a matrix and two vectors associated with it, a
    vector of frequency bins and time bins, where the values in those vectors are the values
    at the bin centers. (As far as vak is concerned, "vector" and "matrix" are synonymous with
    "array".)

    Since both .mat files and .npz files load into a dictionary-like structure,
    the arrays will be accessed with keys. By convention, these keys are 's', 'f', and 't'.
    If you use different keys you can let this function know by changing
    the appropriate arguments: spect_key, freqbins_key, timebins_key
    """
    # pre-conditions ---------------------------------------------------------------------------------------------------
    if spect_format not in validators.VALID_SPECT_FORMATS:
        raise ValueError(
            f"spect_format must be one of '{validators.VALID_SPECT_FORMATS}'; "
            f"format '{spect_format}' not recognized."
        )

    if all([arg is None for arg in (spect_dir, spect_files, spect_annot_map)]):
        raise ValueError('must specify one of: spect_dir, spect_files, spect_annot_map')

    if spect_dir and spect_files:
        raise ValueError('received values for spect_dir and spect_files, unclear which to use')

    if spect_dir and spect_annot_map:
        raise ValueError('received values for spect_dir and spect_annot_map, unclear which to use')

    if spect_files and spect_annot_map:
        raise ValueError('received values for spect_files and spect_annot_map, unclear which to use')

    if annot_list and spect_annot_map:
        raise ValueError(
            'received values for annot_list and spect_annot_map, unclear which annotations to use'
        )

    if labelset is not None:
        if type(labelset) != set:
            raise TypeError(
                f'type of labelset must be set, but was: {type(labelset)}'
            )

    # ---- logging -----------------------------------------------------------------------------------------------------
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    # ---- get a list of spectrogram files + associated annotation files -----------------------------------------------
    if spect_dir:  # then get spect_files from that dir
        # note we already validated format above
        spect_files = glob(os.path.join(spect_dir, f'*{spect_format}'))

    if spect_files:  # (or if we just got them from spect_dir)
        if annot_list:
            spect_annot_map = source_annot_map(spect_files, annot_list)
        else:
            # no annotation, so map spectrogram files to None
            spect_annot_map = dict((spect_path, None)
                                   for spect_path in spect_files)

    # ---- validate spect_annot_map ------------------------------------------------------------------------------------
    # regardless of whether we just made it or user supplied it
    for spect_path, annot in spect_annot_map.items():
        if labelset:  # then assume user wants to filter out files where annotation has labels not in labelset
            labels_set = set(annot.seq.labels)
            # below, set(labels_mapping) is a set of that dict's keys
            if not labels_set.issubset(set(labelset)):
                extra_labels = labels_set - set(labelset)
                # because there's some label in labels
                # that's not in labels_mapping
                logger.info(
                    f'Found labels, {extra_labels}, in {spect_path.name}, '
                    'that are not in labels_mapping. Skipping file.'
                )
                spect_annot_map.pop(spect_path)
                continue

        spect_dict = array_dict_from_path(spect_path, spect_format)

        if spect_key not in spect_dict:
            raise KeyError(
                f"Did not find a spectrogram in file '{spect_path.name}' "
                f"using spect_key '{spect_key}'."
            )

        if 'freq_bins' not in locals() and 'time_bins' not in locals():
            freq_bins = spect_dict[freqbins_key]
            time_bins = spect_dict[timebins_key]
            timebin_dur = timebin_dur_from_vec(time_bins, n_decimals_trunc)
        else:
            if not np.array_equal(spect_dict[freqbins_key], freq_bins):
                raise ValueError(
                    f'freq_bins in {spect_path.name} does not match '
                    'freq_bins from other spectrogram files'
                )
            curr_file_timebin_dur = timebin_dur_from_vec(time_bins,
                                                         n_decimals_trunc)
            if not np.allclose(curr_file_timebin_dur, timebin_dur):
                raise ValueError(
                    f'duration of timebin in file {spect_path.name} did not match '
                    'duration of timebin from other array files.'
                )

        # number of freq. bins should equal number of rows
        if spect_dict[freqbins_key].shape[-1] != spect_dict[spect_key].shape[0]:
            raise ValueError(
                f'length of frequency bins in {spect_path.name} '
                'does not match number of rows in spectrogram'
            )
        # number of time bins should equal number of columns
        if spect_dict[timebins_key].shape[-1] != spect_dict[spect_key].shape[1]:
            raise ValueError(
                f'length of time_bins in {spect_path.name} '
                f'does not match number of columns in spectrogram'
            )

    # ---- actually make the dataframe ---------------------------------------------------------------------------------
    # this is defined here so all other arguments to 'to_dataframe' are in scope
    def _to_record(spect_annot_tuple):
        """helper function that enables parallelized creation of "records",
        i.e. rows for dataframe, from .
        Accepts a two-element tuple containing (1) a dictionary that represents a spectrogram
        and (2) annotation for that file"""
        spect_path, annot = spect_annot_tuple
        spect_dict = array_dict_from_path(spect_path, spect_format)

        spect_dur = spect_dict[spect_key].shape[-1] * timebin_dur
        if audio_path_key in spect_dict:
            audio_path = spect_dict[audio_path_key]
            if type(audio_path) == np.ndarray:
                # (because everything stored in .npz has to be in an ndarray)
                audio_path = audio_path.tolist()
        else:
            # try to figure out audio filename programmatically
            # if we can't, then we'll get back a None
            # (or an error)
            audio_path = find_audio_fname(spect_path)

        if annot is not None:
            # TODO: change to annot.annot_path when changing dependency to crowsetta>=2.0
            annot_path = annot.annot_file
        else:
            annot_path = None

        def abspath(a_path):
            if a_path is None:
                return
            else:
                return str(Path(a_path).absolute())


        record = tuple([
            abspath(audio_path),
            abspath(spect_path),
            abspath(annot_path),
            annot_format,
            spect_dur,
            timebin_dur,
        ])
        return record

    spect_path_annot_tuples = db.from_sequence(spect_annot_map.items())
    logger.info('creating dataset')
    with ProgressBar():
        records = list(spect_path_annot_tuples.map(_to_record))

    return pd.DataFrame.from_records(data=records, columns=DF_COLUMNS)
