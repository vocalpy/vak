"""utility functions for working with audio formats"""
from evfuncs import load_cbin
from scipy.io import wavfile

from ..util.general import _files_from_dir


def swap_return_tuple_elements(func):
    def new_f(*args, **kwargs):
        return_tuple = func(*args, **kwargs)
        return return_tuple[1], return_tuple[0]
    return new_f


load_cbin = swap_return_tuple_elements(load_cbin)


AUDIO_FORMAT_FUNC_MAP = {
    'cbin': load_cbin,
    'wav': wavfile.read
}


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
    if audio_format not in AUDIO_FORMAT_FUNC_MAP:
        raise ValueError(f"'{audio_format}' is not a valid audio format")
    audio_files = _files_from_dir(audio_dir, audio_format)
    return audio_files
