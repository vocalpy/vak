import os
import logging

import numpy as np
import dask.bag as db
from dask.diagnostics import ProgressBar

from ..config import validators
from ..utils.general import _files_from_dir
from ..utils.spect import spectrogram


AUDIO_FORMAT_FUNC_MAP = validators.AUDIO_FORMAT_FUNC_MAP


def files_from_dir(audio_dir, audio_format):
    """get all audio files of a given format
    from a directory or its sub-directories,
    using the file extension associated with that annotation format.

    Parameters
    ----------
    audio_dir : str
        path to directory containing audio files.
    audio_format : str
        valid audio file format. One of {'wav', 'cbin'}.

    Returns
    -------
    audio_files : list
        of paths to audio files
    """
    if audio_format not in validators.VALID_AUDIO_FORMATS:
        raise ValueError(f"'{audio_format}' is not a valid audio format")
    audio_files = _files_from_dir(audio_dir, audio_format)
    return audio_files


def to_arr_files(audio_format,
                 spect_params,
                 output_dir,
                 audio_dir=None,
                 audio_files=None,
                 annot_list=None,
                 audio_annot_map=None,
                 labelset=None,
                 skip_files_with_labels_not_in_labelset=True,
                 freqbins_key='f',
                 timebins_key='t',
                 spect_key='s'):
    """makes spectrograms from audio files and save in array files

    Parameters
    ----------
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}
    audio_dir : str
        path to directory containing audio files from which to make spectrograms
    audio_files : list
        of str, full paths to audio files from which to make spectrograms
    annot_list : list
        of annotations for array files. Default is None.
    audio_annot_map : dict
        Where keys are paths to array files and value corresponding to each key is
        the annotation for that array file.
        Default is None.
    spect_params : dict
        parameters for computing spectrogram, from .ini file
    output_dir : str
        directory in which to save .spect file generated for each .cbin file,
        as described below
    labelset : list
        of str or int, set of unique labels for vocalizations.
    skip_files_with_labels_not_in_labelset : bool
        if True, skip .cbin files where the 'labels' array in the corresponding
        .cbin.not.mat file contains str labels not found in labels_mapping
    freqbins_key : str
        key for accessing vector of frequency bins in files. Default is 'f'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.

    Returns
    -------
    spects_used_path : str
        Full path to file called 'spect_files'
        which contains a list of three-element tuples:
            spect_filename : str, filename of `.spect` file
            spect_dur : float, duration of the spectrogram from cbin
            labels : str, labels from .cbin.not.mat associated with .cbin
                     (string labels for syllables in spectrogram)
        Used when building data sets of a specific duration.

    For each .wav or .cbin filename in the list, a '.spect' file is saved.
    Each '.spect' file contains a "pickled" Python dictionary
    with the following key, value pairs:
        spect : ndarray
            spectrogram
        freq_bins : ndarray
            vector of centers of frequency bins from spectrogram
        time_bins : ndarray
            vector of centers of tme bins from spectrogram
        labeled_timebins : ndarray
            same length as time_bins, but value of each element is a label
            corresponding to that time bin
    """
    if audio_format not in validators.VALID_AUDIO_FORMATS:
        raise ValueError(
            f"audio format must be one of '{validators.VALID_AUDIO_FORMATS}'; "
            f"format '{audio_format}' not recognized."
        )

    if audio_dir and audio_files:
        raise ValueError('received values for audio_dir and audio_files, unclear which to use')

    if audio_dir and audio_annot_map:
        raise ValueError('received values for audio_dir and audio_annot_map, unclear which to use')

    if audio_files and audio_annot_map:
        raise ValueError('received values for audio_files and audio_annot_map, unclear which to use')

    if annot_list and audio_annot_map:
        raise ValueError(
            'received values for annot_list and array_annot_map, unclear which annotations to use'
        )

    if labelset is None and skip_files_with_labels_not_in_labelset is True:
        raise ValueError(
            "must provide labelset when 'skip_files_with_labels_not_in_labelset' is True"
        )

    # validate audio files if supplied by user
    if audio_files is not None:
        # make sure audio files are all the same type, and the same as audio format specified
        exts = []
        for audio_file in audio_files:
            root, ext = os.path.splitext(audio_file)
            exts.append(ext)
        uniq_ext = set(exts)
        if len(uniq_ext) > 1:
            raise ValueError(
                'audio_files should all have the same extension, '
                f'but found more than one: {uniq_ext}'
                )
        else:
            ext_str = uniq_ext.pop()
            if audio_format not in ext_str:
                raise ValueError(
                    f"audio format. '{audio_format}', does not match extensions in audio_files, '{ext_str}''"
                )

    # otherwise get audio files using audio dir (won't need to validate)
    if audio_dir is not None:
        audio_files = files_from_dir(audio_dir, audio_format)

    if audio_annot_map is None:
        # annot_list can be None when creating spectrograms from
        # unlabeled audio for predicting labels
        if annot_list is None:
            # this makes a list of empty tuples to pair with audio files
            annot_list = [() for _ in range(len(audio_files))]

        audio_annot_map = dict(
            (audio_file, annot) for audio_file, annot in zip(audio_files, annot_list)
        )

    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    # this is defined here so all other arguments to 'to_arr_files' are in scope
    def _array_file_from_audio_annot_tup(audio_annot_tup):
        """helper function that enables parallelized creation of array files containing spectrograms.
        Accepts a tuple with the path to an audio file and annotations,
        and returns a Vocalization object."""
        (audio_file, annot) = audio_annot_tup  # tuple unpacking
        basename = os.path.basename(audio_file)

        if skip_files_with_labels_not_in_labelset:
            annot_labelset = set(annot.labels)
            # below, set(labels_mapping) is a set of that dict's keys
            if not annot_labelset.issubset(set(labelset)):
                # because there's some label in labels that's not in labelset
                logger.info(
                    f'found labels in {basename} not in labels_mapping, skipping file'
                )
                return

        fs, dat = AUDIO_FORMAT_FUNC_MAP[audio_format](audio_file)

        s, f, t = spectrogram(dat, fs, **spect_params)

        spect_dict = {spect_key: s,
                      freqbins_key: f,
                      timebins_key: t}

        npz_fname = os.path.join(os.path.normpath(output_dir),
                                 basename + '.spect.npz')
        np.savez(npz_fname, **spect_dict)
        return npz_fname

    bag = db.from_sequence(audio_annot_map.items())
    logger.info('creating array files with spectrograms')
    with ProgressBar():
        array_files = list(bag.map(_array_file_from_audio_annot_tup))

    return array_files
