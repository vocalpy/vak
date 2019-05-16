import os
from glob import glob
import logging

import numpy as np
from scipy.io import loadmat
import dask.bag as db
from dask.diagnostics import ProgressBar

from .classes import Spectrogram, Vocalization, VocalizationDataset
from ..config import validators
from ..utils.general import timebin_dur_from_vec


def from_files(spect_format,
               spect_dir=None,
               spect_files=None,
               annot_list=None,
               spect_annot_map=None,
               labelset=None,
               skip_files_with_labels_not_in_labelset=False,
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
        format of array files. One of {'mat', 'npz'}
    spect_dir : str
        path to directory of files containing spectrograms as arrays.
        Default is None.
    spect_files : list
        List of paths to array files. Default is None.
    annot_list : list
        of annotations for array files. Default is None.
    spect_annot_map : dict
        Where keys are paths to array files and value corresponding to each key is
        the annotation for that array file.
        Default is None.
    labelset : list
        of str or int, set of unique labels for vocalizations.
    skip_files_with_labels_not_in_labelset : bool
        if True, skip array files where the associated annotations contain labels not in labelset.
        Default is False.
    load_spects : bool
        if True, load spectrograms. If False, return a VocalDataset without spectograms loaded.
        Default is True. Set to False when you want to create a VocalDataset for use
        later, but don't want to load all the spectrograms into memory yet.
    n_decimals_trunc : int
        number of decimal places to keep when truncating the timebin duration calculated from
        the spectrogram arrays.
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
            f"array format must be one of '{validators.VALID_SPECT_FORMATS}'; "
            f"format '{spect_format}' not recognized."
        )

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

    if labelset is None and skip_files_with_labels_not_in_labelset is True:
        raise ValueError(
            "must provide labelset when 'skip_files_with_labels_not_in_labelset' is True"
        )

    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    if spect_dir:
        if spect_format == 'mat':
            spect_files = glob(os.path.join(spect_dir, '*.mat'))
        elif spect_format == 'npz':
            spect_files = glob(os.path.join(spect_dir, '*.npz'))

    if spect_files:
        spect_annot_map = dict((arr_path, annot) for arr_path, annot in zip(spect_files, annot_list))

    # this is defined here so all other arguments to 'from_arr_files' are in scope
    def _voc_from_array_annot(arr_path_annot_tup):
        """helper function that enables parallelized creation of list of Vocalizations.
        Accepts a tuple with the path to an array file and annotations,
        and returns a Vocalization object."""
        (arr_path, annot) = arr_path_annot_tup
        arr_file = os.path.basename(arr_path)
        if spect_format == 'mat':
            arr = loadmat(arr_path, squeeze_me=True)
        elif spect_format == 'npz':
            arr = np.load(arr_path)

        if spect_key not in arr:
            logger.info(
                f'Did not find a spectrogram in array file: {arr_file}.\nSkipping this file.\n'
            )
            return

        if skip_files_with_labels_not_in_labelset:
            labels_set = set(annot.labels)
            # below, set(labels_mapping) is a set of that dict's keys
            if not labels_set.issubset(set(labelset)):
                extra_labels = labels_set - set(labelset)
                # because there's some label in labels
                # that's not in labels_mapping
                logger.info(
                    f'Found labels, {extra_labels}, in {arr_file}, that are not in labels_mapping. '
                    'Skipping file.'
                )
                return

        if 'freq_bins' not in locals() and 'time_bins' not in locals():
            freq_bins = arr[freqbins_key]
            time_bins = arr[timebins_key]
            timebin_dur = timebin_dur_from_vec(time_bins, n_decimals_trunc)
        else:
            if not np.array_equal(arr[freqbins_key], freq_bins):
                raise ValueError(
                    f'freq_bins in {arr_file} does not freq_bins from other array files'
                )
            curr_file_timebin_dur = timebin_dur_from_vec(time_bins, n_decimals_trunc)
            if not np.allclose(curr_file_timebin_dur, timebin_dur):
                raise ValueError(
                    f'duration of timebin in file {arr_file} did not match duration of '
                    'timebin from other array files.'
                )

        # number of freq. bins should equal number of rows
        if arr[freqbins_key].shape[-1] != arr[spect_key].shape[0]:
            raise ValueError(
                f'length of frequency bins in {arr_file} does not match number of rows in spectrogram'
            )
        # number of time bins should equal number of columns
        if arr[timebins_key].shape[-1] != arr[spect_key].shape[1]:
            raise ValueError(
                f'length of time_bins in {arr_file} does not match number of columns in spectrogram'
            )

        spect_dur = arr[spect_key].shape[-1] * timebin_dur

        if load_spects:
            spect_dict = {
                'freq_bins': arr[freqbins_key],
                'time_bins': arr[timebins_key],
                'timebin_dur': timebin_dur,
                'array': arr[spect_key],
            }
            spect = Spectrogram(**spect_dict)
        else:
            spect = None

        voc = Vocalization(
            annot=annot,
            spect_file=arr_path,
            spect=spect,
            audio_file=annot.file,
            duration=spect_dur)

        return voc

    arr_path_annot_tups = db.from_sequence(spect_annot_map.items())
    logger.info('creating VocalDataset')
    with ProgressBar():
        voc_list = list(arr_path_annot_tups.map(_voc_from_array_annot))

    return VocalizationDataset(voc_list=voc_list)
