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


def range_str(range_str, sort=True):
    """Generate range of ints from a formatted string,
    then convert range from int to str

    Examples
    --------
    >>> range_str('1-4,6,9-11')
    ['1','2','3','4','6','9','10','11']

    Takes a range in form of "a-b" and returns
    a list of numbers between a and b inclusive.
    Also accepts comma separated ranges like "a-b,c-d,f"  which will
    return a list with numbers from a to b, c to d, and f.

    Parameters
    ----------
    range_str : str
        of form 'a-b,c', where a hyphen indicates a range
        and a comma separates ranges or single numbers
    sort : bool
        If True, sort output before returning. Default is True.

    Returns
    -------
    list_range : list
        of integer values converted to single-character strings, produced by parsing range_str
    """
    # adapted from
    # http://code.activestate.com/recipes/577279-generate-list-of-numbers-from-hyphenated-and-comma/
    s = "".join(range_str.split())  # removes white space
    list_range = []
    for substr in range_str.split(','):
        subrange = substr.split('-')
        if len(subrange) not in [1, 2]:
            raise SyntaxError("unable to parse range {} in labelset {}."
                              .format(subrange, substr))
        list_range.extend(
            [int(subrange[0])]
        ) if len(subrange) == 1 else list_range.extend(
            range(int(subrange[0]), int(subrange[1]) + 1))

    if sort:
        list_range.sort()

    return [str(list_int) for list_int in list_range]


def stripchars(string, chars):
    return string.translate(str.maketrans('', '', chars))
