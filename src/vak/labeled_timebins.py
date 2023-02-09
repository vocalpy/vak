"""functions for dealing with labeled timebin vectors"""
from __future__ import annotations

import numpy as np


def has_unlabeled(labels_int: list | np.nddary,
                  onsets_s: np.ndarray,
                  offsets_s: np.ndarray,
                  time_bins: np.ndarray) -> bool:
    """Determine whether there are unlabeled segments in a spectrogram,
    given labels, onsets, and offsets of segments, and vector of
    time bins from spectrogram.

    Parameters
    ----------
    labels_int : list, numpy.ndarray
        a list or array of labels from the annotation for a vocalization,
        mapped to integers
    onsets_s : numpy.ndarray
        1d vector of floats, segment onsets in seconds
    offsets_s : numpy.ndarray
        1-d vector of floats, segment offsets in seconds
    time_bins : numpy.ndarray
        1-d vector of floats, time in seconds for center of each time bin of a spectrogram

    Returns
    -------
    has_unlabeled : bool
        if True, there are time bins that do not have labels associated with them
    """
    if (
        type(labels_int) == list
        and not all([type(lbl) == int for lbl in labels_int])
        or (
            type(labels_int) == np.ndarray
            and labels_int.dtype not in [np.int8, np.int16, np.int32, np.int64]
        )
    ):
        raise TypeError("labels_int must be a list or numpy.ndarray of integers")

    dummy_unlabeled_label = np.max(labels_int) + 1
    label_vec = np.ones((time_bins.shape[-1], 1), dtype="int8") * dummy_unlabeled_label
    onset_inds = [np.argmin(np.abs(time_bins - onset)) for onset in onsets_s]
    offset_inds = [np.argmin(np.abs(time_bins - offset)) for offset in offsets_s]
    for label, onset, offset in zip(labels_int, onset_inds, offset_inds):
        # offset_inds[ind]+1 because offset time bin is still "part of" syllable
        label_vec[onset : offset + 1] = label

    if dummy_unlabeled_label in label_vec:
        return True
    else:
        return False
