"""Functional forms of transformations
related to frame labels,
i.e., vectors where each element represents
a label for a frame, either a single sample in audio
or a single time bin from a spectrogram.

This module is structured as followed:
- from_segments: transform to get frame labels from annotations
- to_labels: transform to get back just string labels from frame labels,
  used to evaluate a model
- to_segments: transform to get back segment onsets, offsets, and labels from frame labels.
  Inverse of ``from_segments``.
- post-processing transforms that can be used to "clean up" a vector of frame labels
  - segment_inds_list_from_class_labels: helper function used to find segments in a vector of frame labels
  - remove_short_segments: remove any segment less than a minimum duration
  - take_majority_vote: take a "majority vote" within each segment bounded by the background label,
    and apply the most "popular" label within each segment to all timebins in that segment
  - postprocess: combines remove_short_segments and take_majority_vote in one transform
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats

from ... import common
from ...common.timebins import timebin_dur_from_vec
from ...common.validators import column_or_1d, row_or_1d

__all__ = [
    # keep alphabetized
    "boundary_inds_from_boundary_labels",
    "from_segments",
    "postprocess",
    "remove_short_segments",
    "take_majority_vote",
    "segment_inds_list_from_class_labels",
    "to_labels",
    "to_segments",
]


def from_segments(
    labels_int: npt.NDArray,
    onsets_s: npt.NDArray,
    offsets_s: npt.NDArray,
    time_bins: npt.NDArray,
    background_label: int = 0,
) -> npt.NDArray:
    """Make a vector of labels for a vector of frames,
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
    background_label : int
        Label assigned to frames that do not have labels associated with them.
        Default is 0.

    Returns
    -------
    frame_labels : numpy.ndarray
        same length as time_bins, with each element a label for each time bin
    """
    if (
        isinstance(labels_int, list)
        and not all([isinstance(lbl, int) for lbl in labels_int])
    ) or (
        isinstance(labels_int, np.ndarray)
        and labels_int.dtype not in [np.int8, np.int16, np.int32, np.int64]
    ):
        raise TypeError(
            "labels_int must be a list or numpy.ndarray of integers"
        )

    label_vec = (
        np.ones((time_bins.shape[-1],), dtype="int8") * background_label
    )
    onset_inds = [np.argmin(np.abs(time_bins - onset)) for onset in onsets_s]
    offset_inds = [
        np.argmin(np.abs(time_bins - offset)) for offset in offsets_s
    ]
    for label, onset, offset in zip(labels_int, onset_inds, offset_inds):
        # offset_inds[ind]+1 because offset time bin is still "part of" syllable
        label_vec[onset : offset + 1] = label  # noqa: E203

    return label_vec


def to_labels(
    frame_labels: npt.NDArray,
    labelmap: dict,
    background_label: str = common.constants.DEFAULT_BACKGROUND_LABEL,
) -> str:
    """Convert vector of frame labels to a string,
    one character for each continuous segment.

    Allows for converting output of network
    from a label for each frame
    to one label for each continuous segment,
    in order to compute string-based metrics like edit distance.

    Parameters
    ----------
    frame_labels : numpy.ndarray
        A vector where each element represents
        a label for a frame, either a single sample in audio
        or a single time bin from a spectrogram.
        Typically, the output of a neural network.
    labelmap : dict
        That maps string labels to integers.
        The mapping is inverted to convert back to string labels.
    background_label: str, optional
        The string label applied to segments belonging to the
        background class.
        Default is
        :const:`vak.common.constants.DEFAULT_BACKGROUND_LABEL`.

    Returns
    -------
    labels : str
        The label at the onset of each continuous segment
        in ``frame_labels``, mapped back to string labels in ``labelmap``.
    """
    frame_labels = row_or_1d(frame_labels)

    onset_inds = np.diff(frame_labels, axis=0).astype(bool)
    onset_inds = np.insert(onset_inds, 0, True)

    labels = frame_labels[onset_inds]

    # remove background label
    if background_label in labelmap:
        labels = labels[labels != labelmap[background_label]]

    if len(labels) < 1:  # if removing all the 'unlabeled' leaves nothing
        return ""

    # only invert mapping and then map integer labels to characters
    inverse_labelmap = dict((v, k) for k, v in labelmap.items())
    labels = labels.tolist()
    labels = [inverse_labelmap[label] for label in labels]

    return "".join(labels)


def to_segments(
    frame_labels: npt.NDArray,
    labelmap: dict,
    frame_times: npt.NDArray,
    n_decimals_trunc: int = 5,
    background_label: str = common.constants.DEFAULT_BACKGROUND_LABEL,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Convert a vector of frame labels
    into segments in the form of onset indices,
    offset indices, and labels.

    Finds where continuous runs of a single label start
    and stop in timebins, and considers each of these runs
    a segment.

    The function returns vectors of labels and onsets and offsets
    in units of seconds.

    Parameters
    ----------
    frame_labels : numpy.ndarray
        A vector where each element represents
        a label for a frame, either a single sample in audio
        or a single time bin from a spectrogram.
        Output of a neural network.
    labelmap : dict
        That maps labels to consecutive integers.
        The mapping is inverted to convert back to labels.
    frame_times : numpy.ndarray
        Vector of times; the times are either the time of samples in audio,
        or the bin centers of columns in a spectrogram,
        returned by function that generated spectrogram.
        Used to convert onset and offset indices in frame_labels to seconds.
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
    frame_labels = column_or_1d(frame_labels)

    if background_label in labelmap:
        # handle the case when all time bins are predicted to be unlabeled
        # see https://github.com/NickleDave/vak/issues/383
        uniq_frame_labels = np.unique(frame_labels)
        if (
            len(uniq_frame_labels) == 1
            and uniq_frame_labels[0] == labelmap[background_label]
        ):
            return None, None, None

    # used to find onsets/offsets below; compute here so if we fail we do so early
    timebin_dur = timebin_dur_from_vec(frame_times, n_decimals_trunc)

    offset_inds = np.nonzero(np.diff(frame_labels, axis=0))[
        0
    ]  # [0] because nonzero return tuple
    onset_inds = offset_inds + 1
    offset_inds = np.concatenate(
        (offset_inds, np.asarray([frame_labels.shape[0] - 1]))
    )
    onset_inds = np.concatenate((np.asarray([0]), onset_inds))
    labels = frame_labels[onset_inds]

    # remove background label
    if background_label in labelmap:
        keep = np.where(labels != labelmap[background_label])[0]
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
    onsets_s = frame_times[onset_inds] - (timebin_dur / 2)
    offsets_s = frame_times[offset_inds] + (timebin_dur / 2)

    # but this estimate will be "wrong" if we set the onset or offset time
    # outside the possible times in our timebin vector. Need to clean up.
    if onsets_s[0] < 0.0:
        onsets_s[0] = 0.0
    if offsets_s[-1] > frame_times[-1]:
        offsets_s[-1] = frame_times[-1]

    return labels, onsets_s, offsets_s


def segment_inds_list_from_class_labels(
    frame_labels: npt.NDArray, background_label: int = 0
) -> list[npt.NDArray]:
    """Given a vector of frame labels,
    returns a list of indexing vectors,
    one for each segment in the vector
    that is not labeled with the background label.

    Parameters
    ----------
    frame_labels : numpy.ndarray
        A vector where each element represents
        a label for a frame, either a single sample in audio
        or a single time bin from a spectrogram.
    background_label : int
        Label that was given to segments that were not labeled in annotation,
        e.g. silent periods between annotated segments. Default is 0.

    Returns
    -------
    segment_inds_list : list
        Of fancy indexing arrays. Each array can be used to index
        one segment in ``frame_labels``.
    """
    segment_inds = np.nonzero(frame_labels != background_label)[0]
    return np.split(segment_inds, np.where(np.diff(segment_inds) != 1)[0] + 1)


def remove_short_segments(
    frame_labels: npt.NDArray,
    segment_inds_list: list[npt.NDArray],
    timebin_dur: float,
    min_segment_dur: float | int,
    background_label: int = 0,
) -> tuple[npt.NDArray, list[npt.NDArray]]:
    """Remove segments from vector of frame labels
    that are shorter than a specified duration.

    Parameters
    ----------
    frame_labels : numpy.ndarray
        A vector where each element represents
        a label for a frame, either a single sample in audio
        or a single time bin from a spectrogram.
        Output of a neural network.
    segment_inds_list : list
        Of numpy.ndarray, indices that will recover segments list from ``frame_labels``.
        Returned by function ``vak.labels.frame_labels_segment_inds_list``.
    timebin_dur : float
        Duration of a single timebin in the spectrogram, in seconds.
        Used to convert onset and offset indices in ``frame_labels`` to seconds.
    min_segment_dur : float
        Minimum duration of segment, in seconds. If specified, then
        any segment with a duration less than min_segment_dur is
        removed from frame_labels. Default is None, in which case no
        segments are removed.
    background_label : int
        Label that was given to segments that were not labeled in annotation,
        e.g. silent periods between annotated segments. Default is 0.

    Returns
    -------
    frame_labels : numpy.ndarray
        A vector where each element represents
        a label for a frame, either a single sample in audio
        or a single time bin from a spectrogram.
        With segments whose duration is shorter than ``min_segment_dur``
        set to ``background_label``
    segment_inds_list : list
        Of numpy.ndarray, with arrays removed that represented
        segments in ``frame_labels`` that were shorter than ``min_segment_dur``.
    """
    new_segment_inds_list = []

    for segment_inds in segment_inds_list:
        if segment_inds.shape[-1] * timebin_dur < min_segment_dur:
            frame_labels[segment_inds] = background_label
            # DO NOT keep segment_inds array
        else:
            # do keep segment_inds array, don't change frame_labels
            new_segment_inds_list.append(segment_inds)

    return frame_labels, new_segment_inds_list


def take_majority_vote(
    frame_labels: npt.NDArray, segment_inds_list: list[npt.NDArray]
) -> npt.NDArray:
    """Transform segments containing multiple labels
    into segments with a single label by taking a "majority vote",
    i.e. assign all frames in the segment the most frequently
    occurring label in the segment.

    Parameters
    ----------
    frame_labels : numpy.ndarray
        A vector where each element represents
        a label for a frame, either a single sample in audio
        or a single time bin from a spectrogram.
        Output of a neural network.
    segment_inds_list : list
        Of numpy.ndarray, indices that will recover segments list from frame_labels.
        Returned by function ``vak.labels.frame_labels_segment_inds_list``.

    Returns
    -------
    frame_labels : numpy.ndarray
        After the majority vote transform has been applied.
    """
    for segment_inds in segment_inds_list:
        segment = frame_labels[segment_inds]
        majority = scipy.stats.mode(segment, keepdims=False)[0].item()
        frame_labels[segment_inds] = majority

    return frame_labels


def boundary_inds_from_boundary_labels(
    boundary_labels: npt.NDArray, force_boundary_first_ind: bool = True
) -> npt.NDArray:
    """Return a :class:`numpy.ndarray` with the indices
    of boundaries, given a 1-D vector of boundary labels.

    Parameters
    ----------
    boundary_labels : numpy.ndarray
        Vector of integers ``{0, 1}``, where ``1`` indicates a boundary,
        and ``0`` indicates no boundary.
        Output of a frame classification model trained to classify each
        frame as "boundary" or "no boundary".
    force_boundary_first_ind : bool
        If ``True``, and the first index of ``boundary_labels`` is not classified as a boundary,
        force it to be a boundary.
    """
    boundary_labels = row_or_1d(boundary_labels)
    boundary_inds = np.nonzero(boundary_labels)[0]

    if force_boundary_first_ind:
        if len(boundary_inds) == 0:
            # handle edge case where no boundaries were predicted
            boundary_inds = np.array([0])  # replace with a single boundary, at index 0
        else:
            if boundary_inds[0] != 0:
                # force there to be a boundary at index 0
                np.insert(boundary_inds, 0, 0)

    return boundary_inds


def segment_inds_list_from_boundary_labels(
    boundary_labels: npt.NDArray, force_boundary_first_ind: bool = True
) -> list[npt.NDArray]:
    """Given an array of boundary labels,
    return a list of :class:`numpy.ndarray` vectors,
    each of which can be used to index one segment
    in a vector of frame labels.

    Parameters
    ----------
    boundary_labels : numpy.ndarray
        Vector of integers ``{0, 1}``, where ``1`` indicates a boundary,
        and ``0`` indicates no boundary.
        Output of a frame classification model trained to classify each
        frame as "boundary" or "no boundary".
    force_boundary_first_ind : bool
        If ``True``, and the first index of ``boundary_labels`` is not classified as a boundary,
        force it to be a boundary.

    Returns
    -------
    segment_inds_list : list
        Of fancy indexing arrays. Each array can be used to index
        one segment in ``frame_labels``.
    """
    boundary_inds = boundary_inds_from_boundary_labels(
        boundary_labels, force_boundary_first_ind
    )

    # at the end of `boundary_inds``, insert an imaginary "last" boundary we use just with ``np.arange`` below
    np.insert(boundary_inds, boundary_inds.shape[0], boundary_labels.shape[0])

    segment_inds_list = []
    for start, stop in zip(boundary_inds[:-1], boundary_inds[1:]):
        segment_inds_list.append(np.arange(start, stop))

    return segment_inds_list


def postprocess(
    frame_labels: npt.NDArray,
    timebin_dur: float,
    background_label: int = 0,
    min_segment_dur: float | None = None,
    majority_vote: bool = False,
    boundary_labels: npt.NDArray | None = None,
) -> npt.NDArray:
    """Apply post-processing transformations
    to a vector of frame labels.

    Optional post-processing
    consist of two transforms,
    that both rely on there being a label
    that corresponds to the background class.
    The first removes any segments that are
    shorter than a specified duration,
    by converting labels in those segments to the
    background class label.
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
           frame_labels = remove_short_segments(frame_labels, labelmap, min_segment_dur)
       if majority_vote:
           frame_labels = majority_vote(frame_labels, labelmap)
       return frame_labels

    Parameters
    ----------
    frame_labels : numpy.ndarray
        A vector where each element represents
        a label for a frame, either a single sample in audio
        or a single time bin from a spectrogram.
        Output of a neural network.
    timebin_dur : float
        Duration of a time bin in a spectrogram,
        e.g., as estimated from vector of times
        using ``vak.timebins.timebin_dur_from_vec``.
    background_label : int
        Label that was given to segments that were not labeled in annotation,
        e.g. silent periods between annotated segments. Default is 0.
    min_segment_dur : float
        Minimum duration of segment, in seconds. If specified, then
        any segment with a duration less than min_segment_dur is
        removed from frame_labels. Default is None, in which case no
        segments are removed.
    majority_vote : bool
        If True, transform segments containing multiple labels
        into segments with a single label by taking a "majority vote",
        i.e. assign all time bins in the segment the most frequently
        occurring label in the segment.
        This transform requires either a background label
        or a vector of boundary labels.
        Default is False.
    boundary_labels : numpy.ndarray, optional.
        Vector of integers ``{0, 1}``, where ``1`` indicates a boundary,
        and ``0`` indicates no boundary.
        Output of one head of a frame classification model,
        that has been trained to classify each frame as either
        "boundary" or "no boundary".
        Optional, default is None.
        If supplied, this vector is used to find segments
        before applying post-processing, instead
        of recovering them from ``frame_labels`` using the
        ``background_label``

    Returns
    -------
    frame_labels : numpy.ndarray
        Vector of frame labels after post-processing is applied.
    """
    frame_labels = row_or_1d(frame_labels)
    if boundary_labels is not None:
        boundary_labels = row_or_1d(boundary_labels)

    # handle the case when all time bins are predicted to be unlabeled
    # see https://github.com/NickleDave/vak/issues/383
    uniq_frame_labels = np.unique(frame_labels)
    if (
        len(uniq_frame_labels) == 1
        and uniq_frame_labels[0] == background_label
    ):
        return frame_labels  # -> no need to do any of the post-processing

    if boundary_labels is not None:
        segment_inds_list = segment_inds_list_from_boundary_labels(
            boundary_labels
        )
    else:
        segment_inds_list = segment_inds_list_from_class_labels(
            frame_labels, background_label=background_label
        )

    if min_segment_dur is not None:
        frame_labels, segment_inds_list = remove_short_segments(
            frame_labels,
            segment_inds_list,
            timebin_dur,
            min_segment_dur,
            background_label,
        )
        if len(segment_inds_list) == 0:  # no segments left after removing
            return frame_labels  # -> no need to do any of the post-processing

    if majority_vote:
        frame_labels = take_majority_vote(frame_labels, segment_inds_list)

    return frame_labels
