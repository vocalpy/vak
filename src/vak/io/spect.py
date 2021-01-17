"""function that converts a set of array files (.npz, .mat) containing spectrograms
into a pandas DataFrame that represents a dataset used by ``vak``

the returned DataFrame has columns as specified by vak.io.spect.DF_COLUMNS
"""
from glob import glob
import os
from pathlib import Path

import dask.bag as db
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd

from .. import constants
from .. import files
from ..annotation import source_annot_map
from ..converters import labelset_to_set
from ..logging import log_or_print


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
                 n_decimals_trunc=5,
                 freqbins_key='f',
                 timebins_key='t',
                 spect_key='s',
                 audio_path_key='audio_path',
                 logger=None,
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
    labelset : str, list, set
        of str or int, set of unique labels for vocalizations. Default is None.
        If not None, then files will be skipped where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using ``vak.converters.labelset_to_set``.
        See help for that function for details on how to specify labelset.
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

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by vak.logging.get_logger. Default is None.

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
    if spect_format not in constants.VALID_SPECT_FORMATS:
        raise ValueError(
            f"spect_format must be one of '{constants.VALID_SPECT_FORMATS}'; "
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
        labelset = labelset_to_set(labelset)

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

    # --- filter by labelset -------------------------------------------------------------------------------------------
    if labelset:  # then assume user wants to filter out files where annotation has labels not in labelset
        for spect_path in list(
                spect_annot_map.keys()):  # iterate over keys so we can pop from dict without RuntimeError
            annot = spect_annot_map[spect_path]
            labels_set = set(annot.seq.labels)
            # below, set(labels_mapping) is a set of that dict's keys
            if not labels_set.issubset(set(labelset)):
                extra_labels = labels_set - set(labelset)
                # because there's some label in labels
                # that's not in labels_mapping
                log_or_print(f'Found labels, {extra_labels}, in {Path(spect_path).name}, '
                             'that are not in labels_mapping. Skipping file.',
                             logger=logger, level='info')
                spect_annot_map.pop(spect_path)
                continue

    # ---- validate set of spectrogram files ---------------------------------------------------------------------------
    # regardless of whether we just made it or user supplied it
    spect_paths = list(spect_annot_map.keys())
    files.spect.is_valid_set_of_spect_files(spect_paths,
                                            spect_format,
                                            freqbins_key,
                                            timebins_key,
                                            spect_key,
                                            n_decimals_trunc,
                                            logger=logger)

    # now that we have validated that duration of time bins is consistent across files, we can just open one file
    # to get that time bin duration. This way validation function has no side effects, like returning time bin, and
    # this is still relatively fast compared to looping through all files again
    timebin_dur = files.spect.timebin_dur(spect_paths[0],
                                          spect_format,
                                          timebins_key,
                                          n_decimals_trunc)

    # ---- actually make the dataframe ---------------------------------------------------------------------------------
    # this is defined here so all other arguments to 'to_dataframe' are in scope
    def _to_record(spect_annot_tuple):
        """helper function that enables parallelized creation of "records",
        i.e. rows for dataframe, from .
        Accepts a two-element tuple containing (1) a dictionary that represents a spectrogram
        and (2) annotation for that file"""
        spect_path, annot = spect_annot_tuple
        spect_dict = files.spect.load(spect_path, spect_format)

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
            audio_path = files.spect.find_audio_fname(spect_path)

        if annot is not None:
            # TODO: change to annot.annot_path when changing dependency to crowsetta>=2.0
            annot_path = annot.annot_path
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
            annot_format if annot_format else constants.NO_ANNOTATION_FORMAT,
            spect_dur,
            timebin_dur,
        ])
        return record

    spect_path_annot_tuples = db.from_sequence(spect_annot_map.items())
    log_or_print('creating pandas.DataFrame representing dataset from spectrogram files', logger=logger, level='info')
    with ProgressBar():
        records = list(spect_path_annot_tuples.map(_to_record))

    return pd.DataFrame.from_records(data=records, columns=DF_COLUMNS)
