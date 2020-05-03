"""utility functions used by io sub-package"""
from pathlib import Path

import dask.bag as db
from dask.diagnostics import ProgressBar
import numpy as np

from ..logging import log_or_print
from ..util.general import timebin_dur_from_vec
from ..util.path import array_dict_from_path


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
        instance created by vak.util.logging.get_logger. Default is None.

    Returns
    -------
    returns True if all validation checks pass. If not, an error is raised.
    """
    spect_paths = [Path(spect_path) for spect_path in spect_paths]

    def _validate(spect_path):
        """validates each spectrogram file, then returns frequency bin array
        and duration of time bins, so that those can be validated across all files"""
        spect_dict = array_dict_from_path(spect_path, spect_format)

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


def timebin_dur_from_spect_path(spect_path, spect_format, timebins_key, n_decimals_trunc=5):
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
    spect_dict = array_dict_from_path(spect_path, spect_format)
    time_bins = spect_dict[timebins_key]
    timebin_dur = timebin_dur_from_vec(time_bins, n_decimals_trunc)
    return timebin_dur