"""functional forms of transformations
related to labeled timebins,
i.e., vectors where each element represents
a label for a time bin from a spectrogram.

This module is structured as followed:
- from_segments: transform to get labeled timebins from annotations
- to_labels: transform to get back just string labels from labeled timebins,
  used to evaluate a model
- to_segments: transform to get back segment onsets, offsets, and labels from labeled timebins.
  Inverse of ``from_segments``.
- post-processing transforms that can be used to "clean up" a vector of labeled timebins
  - to_inds_list: helper function used to find segments in a vector of labeled timebins
  - remove_short_segments: remove any segment less than a minimum duration
  - take_majority_vote: take a "majority vote" within each segment bounded by the "unlabeled" label,
    and apply the most "popular" label within each segment to all timebins in that segment
  - postprocess: combines remove_short_segments and take_majority_vote in one transform
"""
from __future__ import annotations

import numpy as np
import scipy.stats

from ...timebins import timebin_dur_from_vec
from ...validators import column_or_1d, row_or_1d


__all__ = [
    # keep alphabetized
    'from_segments',
    'postprocess',
    'remove_short_segments',
    'take_majority_vote',
    'to_inds_list',
    'to_labels',
    'to_segments',
]


def from_segments(labels_int: np.ndarray,
                  onsets_s: np.ndarray,
                  offsets_s: np.ndarray,
                  time_bins: np.ndarray,
                  unlabeled_label: int = 0) -> np.ndarray:
    """Make a vector of labels for a vector of time bins,
    given labeled segments in the form of onset times,
    offset times, and segment labels.

    Parameters
    ----------
    labels_int : list, numpy.ndarray
        A list or array of labels from the annotation for a vocalization,
        mapped to integers
    onsets_s : numpy.ndarray
        1-d vector of floats, segment onsets in seconds.
    offsets_s : numpy.ndarray
        1-d vector of floats, segment offsets in seconds.
    time_bins : numpy.ndarray
        1-d vector of floats, time in seconds for center of each time bin of a spectrogram.
    unlabeled_label : int
        Label assigned to time bins that do not have labels associated with them.
        Default is 0.

    Returns
    -------
    lbl_tb : numpy.ndarray
        same length as time_bins, with each element a label for each time bin
    """
    if (
            (
            type(labels_int) == list
            and not all([type(lbl) == int for lbl in labels_int])
            ) or
            (
            type(labels_int) == np.ndarray
            and labels_int.dtype not in [np.int8, np.int16, np.int32, np.int64]
            )
    ):
        raise TypeError("labels_int must be a list or numpy.ndarray of integers")

    label_vec = np.ones((time_bins.shape[-1],), dtype="int8") * unlabeled_label
    onset_inds = [np.argmin(np.abs(time_bins - onset)) for onset in onsets_s]
    offset_inds = [np.argmin(np.abs(time_bins - offset)) for offset in offsets_s]
    for label, onset, offset in zip(labels_int, onset_inds, offset_inds):
        # offset_inds[ind]+1 because offset time bin is still "part of" syllable
        label_vec[onset:offset + 1] = label

    return label_vec


def to_labels(lbl_tb: np.ndarray, labelmap: dict) -> str:
    """Convert vector of labeled timebins to a string,
    one character for each continuous segment.

    Allows for converting output of network
    from a label for each frame
    to one label for each continuous segment,
    in order to compute string-based metrics like edit distance.

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        Where each element is a label for a frame / time bin.
        Typically, the output of a neural network.
    labelmap : dict
        That maps string labels to integers.
        The mapping is inverted to convert back to string labels.

    Returns
    -------
    labels : str
        The label at the onset of each continuous segment
        in ``lbl_tb``, mapped back to string labels in ``labelmap``.
    """
    lbl_tb = row_or_1d(lbl_tb)

    onset_inds = np.diff(lbl_tb, axis=0).astype(bool)
    onset_inds = np.insert(onset_inds, 0, True)

    labels = lbl_tb[onset_inds]

    # remove 'unlabeled' label
    if "unlabeled" in labelmap:
        labels = labels[labels != labelmap["unlabeled"]]

    if len(labels) < 1:  # if removing all the 'unlabeled' leaves nothing
        return ""

    # only invert mapping and then map integer labels to characters
    inverse_labelmap = dict((v, k) for k, v in labelmap.items())
    labels = labels.tolist()
    labels = [inverse_labelmap[label] for label in labels]

    return "".join(labels)


def to_segments(
        lbl_tb: np.ndarray,
        labelmap: dict,
        t: np.ndarray,
        n_decimals_trunc: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a vector of labeled time bins
    into segments in the form of onset indices,
    offset indices, and labels.

    Finds where continuous runs of a single label start
    and stop in timebins, and considers each of these runs
    a segment.

    The function returns vectors of labels and onsets and offsets
    in units of seconds.

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        Vector of labeled spectrogram time bins, i.e.,
        where each element is a label for a time bin.
        Output of a neural network.
    labelmap : dict
        That maps labels to consecutive integers.
        The mapping is inverted to convert back to labels.
    t : numpy.ndarray
        Vector of times; the times are bin centers of columns in a spectrogram.
        Returned by function that generated spectrogram.
        Used to convert onset and offset indices in lbl_tb to seconds.
    n_decimals_trunc : int
        Number of decimal places to keep when truncating the timebin duration
        calculated from the vector of times t. Default is 5.

    Returns
    -------
    labels : numpy.ndarray
        Vector where each element is a label for a segment with its onset
        and offset indices given by the corresponding element in onset_inds
        and offset_inds.
    onsets_s : numpy.ndarray
        Vector where each element is the onset in seconds a segment.
        Each onset corresponds to the value at the same index in labels.
    offsets_s : numpy.ndarray
        Vector where each element is the offset in seconds of a segment.
        Each offset corresponds to the value at the same index in labels.
    """
    lbl_tb = column_or_1d(lbl_tb)

    if "unlabeled" in labelmap:
        # handle the case when all time bins are predicted to be unlabeled
        # see https://github.com/NickleDave/vak/issues/383
        uniq_lbl_tb = np.unique(lbl_tb)
        if len(uniq_lbl_tb) == 1 and uniq_lbl_tb[0] == labelmap["unlabeled"]:
            return None, None, None

    # used to find onsets/offsets below; compute here so if we fail we do so early
    timebin_dur = timebin_dur_from_vec(t, n_decimals_trunc)

    offset_inds = np.nonzero(np.diff(lbl_tb, axis=0))[0]  # [0] because nonzero return tuple
    onset_inds = offset_inds + 1
    offset_inds = np.concatenate((offset_inds, np.asarray([lbl_tb.shape[0] - 1])))
    onset_inds = np.concatenate((np.asarray([0]), onset_inds))
    labels = lbl_tb[onset_inds]

    # remove 'unlabeled' label
    if "unlabeled" in labelmap:
        keep = np.where(labels != labelmap["unlabeled"])[0]
        labels = labels[keep]
        onset_inds = onset_inds[keep]
        offset_inds = offset_inds[keep]

    # handle case where removing 'unlabeled' leaves no segments
    if all([len(vec) == 0 for vec in (labels, onset_inds, offset_inds)]):
        return None, None, None

    inverse_labelmap = dict((v, k) for k, v in labelmap.items())
    labels = labels.tolist()
    labels = np.asarray([inverse_labelmap[label] for label in labels])
    # the 'best' estimate we can get of onset and offset times,
    # given binned times, and labels applied to each time bin,
    # is "some time" between the last labeled bin for one segment,
    # i.e. its offset, and the first labeled bin for the next
    # segment, i.e. its onset. In other words if the whole bin is labeled
    # as belonging to that segment, and the bin preceding it is labeled as
    # belonging to the previous section, then the onset of the current
    # segment must be the time between the two bins. To find those times
    # we use the bin centers and either subtract (for onsets) or add
    # (for offsets) half a timebin duration. This half a timebin
    # duration puts our onsets and offsets at the time "between" bins.
    onsets_s = t[onset_inds] - (timebin_dur / 2)
    offsets_s = t[offset_inds] + (timebin_dur / 2)

    # but this estimate will be "wrong" if we set the onset or offset time
    # outside the possible times in our timebin vector. Need to clean up.
    if onsets_s[0] < 0.0:
        onsets_s[0] = 0.0
    if offsets_s[-1] > t[-1]:
        offsets_s[-1] = t[-1]

    return labels, onsets_s, offsets_s


def to_inds_list(lbl_tb: np.ndarray, unlabeled_label: int = 0) -> list[np.ndarray]:
    """Given a vector of labeled timebins,
    returns a list of indexing vectors,
    one for each labeled segment in the vector.

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        Vector of labeled timebins from spectrogram
    unlabeled_label : int
        Label that was given to segments that were not labeled in annotation,
        e.g. silent periods between annotated segments. Default is 0.
    return_inds : bool
        If True, return list of indices for segments in lbl_tb, in addition to the segments themselves.
        If False, just return list of numpy.ndarrays that are the segments from lbl_tb.

    Returns
    -------
    segment_inds_list : list
        of numpy.ndarray, indices that will recover segments list from lbl_tb.
    """
    segment_inds = np.nonzero(lbl_tb != unlabeled_label)[0]
    return np.split(segment_inds, np.where(np.diff(segment_inds) != 1)[0] + 1)


def remove_short_segments(
        lbl_tb: np.ndarray,
        segment_inds_list: list[np.ndarray],
        timebin_dur: float,
        min_segment_dur: float | int,
        unlabeled_label: int = 0
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Remove segments from vector of labeled timebins
    that are shorter than a specified duration.

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        Vector of labeled spectrogram time bins, i.e.,
        where each element is a label for a time bin.
        Output of a neural network.
    segment_inds_list : list
        Of numpy.ndarray, indices that will recover segments list from ``lbl_tb``.
        Returned by function ``vak.labels.lbl_tb_segment_inds_list``.
    timebin_dur : float
        Duration of a single timebin in the spectrogram, in seconds.
        Used to convert onset and offset indices in ``lbl_tb`` to seconds.
    min_segment_dur : float
        Minimum duration of segment, in seconds. If specified, then
        any segment with a duration less than min_segment_dur is
        removed from lbl_tb. Default is None, in which case no
        segments are removed.
    unlabeled_label : int
        Label that was given to segments that were not labeled in annotation,
        e.g. silent periods between annotated segments. Default is 0.

    Returns
    -------
    lbl_tb : numpy.ndarray
        With segments whose duration is shorter than ``min_segment_dur``
        set to ``unlabeled_label``
    segment_inds_list : list
        Of numpy.ndarray, with arrays removed that represented
        segments in ``lbl_tb`` that were shorter than ``min_segment_dur``.
    """
    new_segment_inds_list = []

    for segment_inds in segment_inds_list:
        if segment_inds.shape[-1] * timebin_dur < min_segment_dur:
            lbl_tb[segment_inds] = unlabeled_label
            # DO NOT keep segment_inds array
        else:
            # do keep segment_inds array, don't change lbl_tb
            new_segment_inds_list.append(segment_inds)

    return lbl_tb, new_segment_inds_list


def take_majority_vote(lbl_tb: np.ndarray,
                       segment_inds_list: list[np.ndarray]) -> np.ndarray:
    """Transform segments containing multiple labels
    into segments with a single label by taking a "majority vote",
    i.e. assign all time bins in the segment the most frequently
    occurring label in the segment.

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        Vector of labeled spectrogram time bins, i.e.,
        where each element is a label for a time bin.
        Output of a neural network.
    segment_inds_list : list
        Of numpy.ndarray, indices that will recover segments list from lbl_tb.
        Returned by function ``vak.labels.lbl_tb_segment_inds_list``.

    Returns
    -------
    lbl_tb : numpy.ndarray
        After the majority vote transform has been applied.
    """
    for segment_inds in segment_inds_list:
        segment = lbl_tb[segment_inds]
        majority = scipy.stats.mode(segment, keepdims=False)[0].item()
        lbl_tb[segment_inds] = majority

    return lbl_tb


def postprocess(
        lbl_tb: np.ndarray,
        timebin_dur: float,
        unlabeled_label: int = 0,
        min_segment_dur: float | None = None,
        majority_vote: bool = False,
) -> np.ndarray:
    """Apply post-processing transformations
    to a vector of labeled timebins.

    Optional post-processing
    consist of two transforms,
    that both rely on there being a label
    that corresponds to the "unlabeled"
    (or "background") class.
    The first removes any segments that are
    shorter than a specified duration,
    by converting labels in those segments to the
    "background" / "unlabeled" class label.
    The second performs a "majority vote"
    transform within run of labels that is
    bordered on both sides by the "background" label.
    I.e., it counts the number of times any
    label occurs in that segment,
    and then assigns all bins the most common label.

    The function performs those steps in this order
    (pseudo-code):

    .. code-block::

       if min_segment_dur:
           lbl_tb = remove_short_segments(lbl_tb, labelmap, min_segment_dur)
       if majority_vote:
           lbl_tb = majority_vote(lbl_tb, labelmap)
       return lbl_tb

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        Vector of labeled spectrogram time bins, i.e.,
        where each element is a label for a time bin.
        Output of a neural network.
    timebin_dur : float
        Duration of a time bin in a spectrogram,
        e.g., as estimated from vector of times
        using ``vak.timebins.timebin_dur_from_vec``.
    unlabeled_label : int
        Label that was given to segments that were not labeled in annotation,
        e.g. silent periods between annotated segments. Default is 0.
    min_segment_dur : float
        Minimum duration of segment, in seconds. If specified, then
        any segment with a duration less than min_segment_dur is
        removed from lbl_tb. Default is None, in which case no
        segments are removed.
    majority_vote : bool
        If True, transform segments containing multiple labels
        into segments with a single label by taking a "majority vote",
        i.e. assign all time bins in the segment the most frequently
        occurring label in the segment. This transform can only be
        applied if the labelmap contains an 'unlabeled' label,
        because unlabeled segments makes it possible to identify
        the labeled segments. Default is False.

    Returns
    -------
    lbl_tb : numpy.ndarray
        Vector of labeled timebins after post-processing is applied.
    """
    lbl_tb = row_or_1d(lbl_tb)

    # handle the case when all time bins are predicted to be unlabeled
    # see https://github.com/NickleDave/vak/issues/383
    uniq_lbl_tb = np.unique(lbl_tb)
    if len(uniq_lbl_tb) == 1 and uniq_lbl_tb[0] == unlabeled_label:
        return lbl_tb  # -> no need to do any of the post-processing

    segment_inds_list = to_inds_list(
        lbl_tb, unlabeled_label=unlabeled_label
    )

    if min_segment_dur is not None:
        lbl_tb, segment_inds_list = remove_short_segments(
            lbl_tb,
            segment_inds_list,
            timebin_dur,
            min_segment_dur,
            unlabeled_label,
        )
        if len(segment_inds_list) == 0:  # no segments left after removing
            return lbl_tb  # -> no need to do any of the post-processing

    if majority_vote:
        lbl_tb = take_majority_vote(lbl_tb, segment_inds_list)

    return lbl_tb
