from pathlib import Path

import numpy as np
from dask import bag as db
from dask.diagnostics import ProgressBar

from .. import constants
from ..logging import log_or_print
from .files import find_fname
from ..timebins import timebin_dur_from_vec


def find_audio_fname(spect_path, audio_ext=None):
    """finds name of audio file in a path to a spectogram file,
    if one is present.

    Checks for any extension that is a valid audio file format
    and returns path up to and including that extension,
    i.e. with the spectrogram file extension removed.

    Parameters
    ----------
    spect_path : str, Path
        path to a spectrogram file
    audio_ext : str
        extension associated with an audio file format, used to
        find audio file name in spect_path.
        Default is None. If None, search for any valid audio format
        (as defined by vak.config.constants.VALID_AUDIO_FORMATS)

    Returns
    -------
    audio_fname : str
        name of audio file found in spect_path
    """
    if audio_ext is None:
        audio_ext = constants.VALID_AUDIO_FORMATS
    elif type(audio_ext) is str:
        audio_ext = [audio_ext]
    else:
        raise TypeError(
            f'invalid type for audio_ext: {type(audio_ext)}'
        )

    audio_fnames = []
    for ext in audio_ext:
        audio_fnames.append(
            find_fname(str(spect_path), ext)
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


def load(spect_path, spect_format=None):
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
    spect_dict = constants.SPECT_FORMAT_LOAD_FUNCTION_MAP[spect_format](spect_path)
    return spect_dict


def timebin_dur(spect_path, spect_format, timebins_key, n_decimals_trunc=5):
    """get duration of time bins from a spectrogram file

    Parameters
    ----------
    spect_path: str, Path
        path to spectrogram file.
    spect_format : str
        format of file containing spectrogram. One of {'mat', 'npz'}
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    n_decimals_trunc : int
        number of decimal places to keep when truncating the timebin duration calculated from
        the vector of time bins.
        Default is 3, i.e. assumes milliseconds is the last significant digit.

    Returns
    -------
    timebin_dur : float

    """
    spect_path = Path(spect_path)
    spect_dict = load(spect_path, spect_format)
    time_bins = spect_dict[timebins_key]
    timebin_dur = timebin_dur_from_vec(time_bins, n_decimals_trunc)
    return timebin_dur


def is_valid_set_of_spect_files(spect_paths,
                                spect_format,
                                freqbins_key='f',
                                timebins_key='t',
                                spect_key='s',
                                n_decimals_trunc=5,
                                logger=None
                                ):
    """validate a set of spectrogram files that will be used as a dataset.
    Validates that:
      - all files contain a spectrogram array that can be accessed with the specified key
      - the length of the frequency bin array in each file equals the number of rows in the spectrogram array
      - the frequency bins are the same across all files
      - the length of the time bin array in each file equals the number of columns in the spectrogram array
      - the duration of a spectrogram time bin is the same across all files

    Parameters
    ----------
    spect_paths: list
        of strings or pathlib.Path objects; paths to spectrogram files.
    spect_format : str
        format of files containing spectrograms. One of {'mat', 'npz'}
    freqbins_key : str
        key for accessing vector of frequency bins in files. Default is 'f'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    n_decimals_trunc : int
        number of decimal places to keep when truncating the timebin duration calculated from
        the vector of time bins.
        Default is 3, i.e. assumes milliseconds is the last significant digit.

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by vak.logging.get_logger. Default is None.

    Returns
    -------
    returns True if all validation checks pass. If not, an error is raised.
    """
    spect_paths = [Path(spect_path) for spect_path in spect_paths]

    def _validate(spect_path):
        """validates each spectrogram file, then returns frequency bin array
        and duration of time bins, so that those can be validated across all files"""
        spect_dict = load(spect_path, spect_format)

        if spect_key not in spect_dict:
            raise KeyError(
                f"Did not find a spectrogram in file '{spect_path.name}' "
                f"using spect_key '{spect_key}'."
            )

        freq_bins = spect_dict[freqbins_key]
        time_bins = spect_dict[timebins_key]
        timebin_dur = timebin_dur_from_vec(time_bins, n_decimals_trunc)

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

        return spect_path, freq_bins, timebin_dur

    spect_paths_bag = db.from_sequence(spect_paths)

    log_or_print('validating set of spectrogram files', logger=logger, level='info')

    with ProgressBar():
        path_freqbins_timebin_dur_tups = list(spect_paths_bag.map(_validate))

    all_freq_bins = np.stack(
        [tup[1] for tup in path_freqbins_timebin_dur_tups]
    )
    uniq_freq_bins = np.unique(all_freq_bins, axis=0)
    if len(uniq_freq_bins) != 1:
        raise ValueError(
            f'Found more than one frequency bin vector across files. '
            f'Instead found {len(uniq_freq_bins)}'
        )

    timebin_durs = [tup[2] for tup in path_freqbins_timebin_dur_tups]
    uniq_durs = np.unique(timebin_durs)
    if len(uniq_durs) != 1:
        raise ValueError(
            'Found more than one duration for time bins across spectrogram files. '
            f'Durations found were: {uniq_durs}'
        )

    return True
