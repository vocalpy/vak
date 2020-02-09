"""utility functions that deal with paths"""
from functools import partial
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..config import validators
from .general import find_fname


def find_audio_fname(spect_path, audio_ext=None):
    """finds name of audio file in a path to a spectogram file,
    if one is present.

    Checks for any extension that is a valid audio file format
    and returns path up to and including that extension,
    i.e. with the spectrogram file extension removed.

    Parameters
    ----------
    spect_path : str
        path to a spectrogram file
    audio_ext : str
        extension associated with an audio file format, used to
        find audio file name in spect_path.
        Default is None. If None, search for any valid audio format
        (as defined by vak.config.validators.VALID_AUDIO_FORMATS)

    Returns
    -------
    audio_fname : str
        name of audio file found in spect_path
    """
    if audio_ext is None:
        audio_ext = validators.VALID_AUDIO_FORMATS
    elif type(audio_ext) is str:
        audio_ext = [audio_ext]
    else:
        raise TypeError(
            f'invalid type for audio_ext: {type(audio_ext)}'
        )

    audio_fnames = []
    for ext in audio_ext:
        audio_fnames.append(
            find_fname(spect_path, ext)
        )
    # remove Nones
    audio_fnames = [path for path in audio_fnames if path is not None]
    # keep just file name from spect path
    audio_fnames = [Path(path).name for path in audio_fnames]

    if len(audio_fnames) == 1:
        return audio_fnames[0]
    else:
        raise ValueError(
            f'unable to determine filename of audio file from: {spect_path}'
        )


SPECT_FORMAT_LOAD_FUNCTION_MAP = {
    'mat': partial(loadmat, squeeze_me=True),
    'npz': np.load,
}


def array_dict_from_path(spect_path, spect_format=None):
    """load spectrogram and related arrays from a file,
    return as an object that provides Python dictionary-like
    access

    Parameters
    ----------
    spect_path : str, Path
        to an array file.
    spect_format : str
        Valid formats are defined in vak.io.spect.SPECT_FORMAT_LOAD_FUNCTION_MAP.
        Default is None, in which case the extension of the file is used.

    Returns
    -------
    spect_dict : dict-like
        either a dictionary or dictionary-like object that provides access to arrays
        from the file via keys, e.g. spect_dict['s'] for the spectrogram.
        See docstring for vak.audio.to_spect for default keys for spectrogram
        array files that function creates.
    """
    spect_path = Path(spect_path)
    if spect_format is None:
        # "replace('.', '')", because suffix returns file extension with period included
        spect_format = spect_path.suffix.replace('.', '')
    spect_dict = SPECT_FORMAT_LOAD_FUNCTION_MAP[spect_format](spect_path)
    return spect_dict
