"""module for functions that deal with vector of times from a spectrogram,
i.e. where elements are the times at bin centers"""
import numpy as np


def timebin_dur_from_vec(time_bins, n_decimals_trunc=5):
    """compute duration of a time bin, given the
    vector of time bin centers associated with a spectrogram

    Parameters
    ----------
    time_bins : numpy.ndarray
        vector of times in spectrogram, where each value is a bin center.
    n_decimals_trunc : int
        number of decimal places to keep when truncating the timebin duration calculated from
        the spectrogram arrays. Default is 5.

    Returns
    -------
    timebin_dur : float
        duration of a timebin, estimated from vector of times

    Notes
    -----
    takes mean of pairwise difference between neighboring time bins,
    to deal with floating point error, then rounds and truncates to specified decimal place
    """
    # first we round to the given number of decimals
    timebin_dur = np.around(
        np.mean(np.diff(time_bins)),
        decimals=n_decimals_trunc
    )
    # only after rounding do we truncate any decimal place past decade
    decade = 10 ** n_decimals_trunc
    timebin_dur = np.trunc(timebin_dur * decade) / decade
    return timebin_dur
