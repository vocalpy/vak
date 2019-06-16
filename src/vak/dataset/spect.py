import os
from glob import glob
import logging

import numpy as np
from scipy.io import loadmat
import dask.bag as db
from dask.diagnostics import ProgressBar

from .annotation import source_annot_map
from .classes import MetaSpect, Vocalization, VocalizationDataset
from ..config import validators
from ..utils.general import timebin_dur_from_vec


def from_files(spect_format,
               spect_dir=None,
               spect_files=None,
               annot_list=None,
               spect_annot_map=None,
               labelset=None,
               load_spects=True,
               n_decimals_trunc=3,
               freqbins_key='f',
               timebins_key='t',
               spect_key='s'
               ):
    """create VocalizationDataset from already-made spectrograms that are in
    files containing arrays, i.e., .mat files created by Matlab or .npz files created by numpy

    Each file should contain a spectrogram as a matrix and two vectors associated with it, a
    vector of frequency bins and time bins, where the values in those vectors are the values
    at the bin centers. (As far as vak is concerned, "vector" and "matrix" are synonymous with
    "array".)

    Since both .mat files and .npz files load into a dictionary-like structure,
    the arrays will be accessed with keys. By convention, these keys are 's', 'f', and 't'.
    If you use different keys you can let this function know by changing
    the appropriate arguments: spect_key, freqbins_key, timebins_key

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
        of annotations for array files. Default is None.
    spect_annot_map : dict
        Where keys are paths to files and value corresponding to each key is
        the annotation for that file.
        Default is None.
    labelset : list
        of str or int, set of unique labels for vocalizations. Default is None.
        If not None, skip files where the associated annotations contain labels not in labelset.
    load_spects : bool
        if True, load spectrograms. If False, return a VocalDataset without spectograms loaded.
        Default is True. Set to False when you want to create a VocalDataset for use
        later, but don't want to load all the spectrograms into memory yet.
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

    Returns
    -------
    vocalset : vak.dataset.VocalDataset
        dataset of annotated vocalizations
    """
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

    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    if spect_dir:  # then get spect_files from that dir
        if spect_format == 'mat':
            spect_files = glob(os.path.join(spect_dir, '*.mat'))
        elif spect_format == 'npz':
            spect_files = glob(os.path.join(spect_dir, '*.npz'))

    if spect_files:  # (or if we just got them from spect_dir)
        if annot_list:
            spect_annot_map = source_annot_map(spect_files, annot_list)
        else:
            # map spectrogram files to None
            spect_annot_map = dict((spect_path, None)
                                   for spect_path in spect_files)

    # lastly need to validate spect_annot_map
    # regardless of whether we just made it or user supplied it
    for spect_path, annot in spect_annot_map.items():
        # get just file name so error messages don't have giant path
        spect_file = os.path.basename(spect_path)

        if labelset:
            labels_set = set(annot.labels)
            # below, set(labels_mapping) is a set of that dict's keys
            if not labels_set.issubset(set(labelset)):
                extra_labels = labels_set - set(labelset)
                # because there's some label in labels
                # that's not in labels_mapping
                logger.info(
                    f'Found labels, {extra_labels}, in {spect_file}, '
                    'that are not in labels_mapping. Skipping file.'
                )
                spect_annot_map.pop(spect_path)
                continue

        if spect_format == 'mat':
            spect_dict = loadmat(spect_path, squeeze_me=True)
        elif spect_format == 'npz':
            spect_dict = np.load(spect_path)

        if spect_key not in spect_dict:
            raise KeyError(
                f"Did not find a spectrogram in file '{spect_file}' "
                f"using spect_key '{spect_key}'."
            )

        if 'freq_bins' not in locals() and 'time_bins' not in locals():
            freq_bins = spect_dict[freqbins_key]
            time_bins = spect_dict[timebins_key]
            timebin_dur = timebin_dur_from_vec(time_bins, n_decimals_trunc)
        else:
            if not np.array_equal(spect_dict[freqbins_key], freq_bins):
                raise ValueError(
                    f'freq_bins in {spect_file} does not match '
                    'freq_bins from other spectrogram files'
                )
            curr_file_timebin_dur = timebin_dur_from_vec(time_bins,
                                                         n_decimals_trunc)
            if not np.allclose(curr_file_timebin_dur, timebin_dur):
                raise ValueError(
                    f'duration of timebin in file {spect_file} did not match '
                    'duration of timebin from other array files.'
                )

        # number of freq. bins should equal number of rows
        if spect_dict[freqbins_key].shape[-1] != spect_dict[spect_key].shape[0]:
            raise ValueError(
                f'length of frequency bins in {spect_file} '
                'does not match number of rows in spectrogram'
            )
        # number of time bins should equal number of columns
        if spect_dict[timebins_key].shape[-1] != spect_dict[spect_key].shape[1]:
            raise ValueError(
                f'length of time_bins in {spect_file} '
                f'does not match number of columns in spectrogram'
            )

    # this is defined here so all other arguments to 'from_arr_files' are in scope
    def _voc_from_spect_path_annot_tup(spect_path_annot_tup):
        """helper function that enables parallelized creation of list of Vocalizations.
        Accepts a tuple with the path to an spectrogram file and annotations,
        and returns a Vocalization object."""
        (spect_path, annot) = spect_path_annot_tup
        if spect_format == 'mat':
            spect_dict = loadmat(spect_path, squeeze_me=True)
        elif spect_format == 'npz':
            spect_dict = np.load(spect_path)

        spect_dur = spect_dict[spect_key].shape[-1] * timebin_dur

        if load_spects:
            spect_kwargs = {
                'freq_bins': spect_dict[freqbins_key],
                'time_bins': spect_dict[timebins_key],
                'timebin_dur': timebin_dur,
                'spect': spect_dict[spect_key],
            }
            metaspect = MetaSpect(**spect_kwargs)
        else:
            metaspect = None

        voc_kwargs = {
            'annot': annot,
            'spect_path': spect_path,
            'metaspect': metaspect,
            'duration': spect_dur
        }
        if annot is not None:
            voc_kwargs['audio_path'] = annot.file

        return Vocalization(**voc_kwargs)

    spect_path_annot_tups = db.from_sequence(spect_annot_map.items())
    logger.info('creating VocalizationDataset')
    with ProgressBar():
        voc_list = list(spect_path_annot_tups.map(_voc_from_spect_path_annot_tup))
    return VocalizationDataset(voc_list=voc_list, labelset=labelset)
