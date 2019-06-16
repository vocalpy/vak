import os
from glob import glob

import numpy as np


def _files_from_dir(dir_path, ext):
    """helper function that gets all files with a given extension
    from a directory or its sub-directories.

    If no files with the specified extension are found in the directory, then
    the function recurses into all sub-directories and returns any files with
    the extension in those sub-directories.

    Parameters
    ----------
    dir_path : str
        path to target directory
    ext : str
        file extension to search for

    Returns
    -------
    files : list
        of paths to files with specified file extension

    Notes
    -----
    used by vak.dataset.audio.files_from_dir and vak.dataset.annot.files_from_dir
    """
    wildcard_with_extension = f'*.{ext}'
    files = sorted(
        glob(os.path.join(dir_path, wildcard_with_extension))
    )
    if len(files) == 0:
        # if we don't any files with extension, look in sub-directories
        files = []
        subdirs = glob(os.path.join(dir_path, '*/'))
        for subdir in subdirs:
            files.extend(
                glob(os.path.join(dir_path, subdir, wildcard_with_extension))
            )

    if len(files) == 0:
        raise FileNotFoundError(
            f'No files with extension {ext} found in '
            f'{dir_path} or immediate sub-directories'
        )

    return files


def timebin_dur_from_vec(time_bins, n_decimals_trunc=3):
    """compute duration of a time bin, given the
    vector of time bin centers associated with a spectrogram

    Parameters
    ----------
    time_bins : numpy.ndarray
        vector of times in spectrogram, where each value is a bin center.
    n_decimals_trunc : int
        number of decimal places to keep when truncating the timebin duration calculated from
        the spectrogram arrays.
        Default is 3, i.e. assumes milliseconds is the last significant digit.

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


def safe_truncate(X, Y, spect_ID_vector, labelmap, target_dur, timebin_dur):
    correct_length = np.round(target_dur / timebin_dur).astype(int)
    if X.shape[-1] == correct_length:
        return X, Y, spect_ID_vector
    elif X.shape[-1] > correct_length:
        # truncate from the front
        X_out = X[:, :correct_length]
        Y_out = Y[:correct_length, :]
        spect_ID_vector_out = spect_ID_vector[:correct_length]

        mapvals = np.asarray(
            sorted(list(labelmap.values()))
        )

        if np.array_equal(np.unique(Y_out), mapvals):
            pass
        else:
            # if all classes weren't in truncated arrays,
            # try truncating from the back instead
            X_out = X[:, -correct_length:]
            Y_out = Y[-correct_length:, :]
            spect_ID_vector_out = spect_ID_vector[:correct_length]

            if not np.array_equal(np.unique(Y_out), mapvals):
                raise ValueError(
                    "was not able to truncate in a way that maintained all classes in dataset"
                )

        return X_out, Y_out, spect_ID_vector_out

    elif X.shape[-1] < correct_length:
        raise ValueError(
            f"arrays have length {X.shape[-1]} that is shorter than correct length, {correct_length}, "
            f"(= target duration {target_dur} / duration of timebins, {timebin_dur})."
        )
