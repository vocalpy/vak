"""class forms of transformations
related to labeled timebins,
i.e., vectors where each element represents
a label for a time bin from a spectrogram.

These classes call functions from
``vak.transforms.labeled_timebins.functional``.
Not all functions in that module
have a corresponding class,
just key functions needed by
dataloaders and models.

- FromSegments: transform to get labeled timebins from annotations
- ToLabels: transform to get back just string labels from labeled timebins,
  used to evaluate a model.
- ToSegments: transform to get segment onsets, offsets, and labels from labeled timebins.
    Used to convert model output to predictions.
    Inverse of ``from_segments``.
- PostProcess: combines two post-processing transforms applied to labeled timebins,
  ``remove_short_segments`` and ``take_majority_vote``, in one class.
"""
from __future__ import annotations

import numpy as np

from . import functional as F


class FromSegments:
    """Transform that makes a vector of labels for a vector of time bins,
    given labeled segments in the form of onset times,
    offset times, and segment labels.

    Attributes
    ----------
    unlabeled_label : int
        Label assigned to time bins that do not have labels associated with them.
        Default is 0.
    """
    def __init__(self, unlabeled_label: int = 0):
        self.unlabeled_label = unlabeled_label

    def __call__(self,
                 labels_int: np.ndarray,
                 onsets_s: np.ndarray,
                 offsets_s: np.ndarray,
                 time_bins: np.ndarray) -> np.ndarray:
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

        Returns
        -------
        lbl_tb : numpy.ndarray
            same length as time_bins, with each element a label for each time bin
        """
        return F.from_segments(labels_int, onsets_s, offsets_s, time_bins,
                               unlabeled_label=self.unlabeled_label)


class ToLabels:
    """Transforms that converts
    vector of labeled timebins to a string,
    one character for each continuous segment.

    Allows for converting output of network
    from a label for each frame
    to one label for each continuous segment,
    in order to compute string-based metrics like edit distance.

    Attributes
    ----------
    labelmap : dict
        That maps string labels to integers.
        The mapping is inverted to convert back to string labels.
    """
    def __init__(self, labelmap: dict):
        self.labelmap = labelmap

    def __call__(self, lbl_tb: np.ndarray) -> str:
        """Convert vector of labeled timebins to a string,
        one character for each continuous segment.

        Parameters
        ----------
        lbl_tb : numpy.ndarray
            Where each element is a label for a frame / time bin.
            Typically, the output of a neural network.

        Returns
        -------
        labels : str
            The label at the onset of each continuous segment
            in ``lbl_tb``, mapped back to string labels in ``labelmap``.
        """
        return F.to_labels(lbl_tb, self.labelmap)


class ToSegments:
    """Transform that converts a vector of labeled time bins
    into segments in the form of onset indices,
    offset indices, and labels.

    Finds where continuous runs of a single label start
    and stop in timebins, and considers each of these runs
    a segment.

    The function returns vectors of labels and onsets and offsets
    in units of seconds.

    Attributes
    ----------
    labelmap : dict
        That maps string labels to integers.
        The mapping is inverted to convert back to string labels.
    n_decimals_trunc : int
        Number of decimal places to keep when truncating the timebin duration
        calculated from the vector of times t. Default is 5.
    """

    def __init__(self,
                 labelmap: dict,
                 n_decimals_trunc: int = 5
                 ):
        self.labelmap = labelmap
        self.n_decimals_trunc = n_decimals_trunc

    def __call__(self,
                 lbl_tb: np.ndarray,
                 t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        t : numpy.ndarray
            Vector of times; the times are bin centers of columns in a spectrogram.
            Returned by function that generated spectrogram.
            Used to convert onset and offset indices in lbl_tb to seconds.

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
        return F.to_segments(lbl_tb, self.labelmap, t, self.n_decimals_trunc)


class PostProcess:
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

    Attributes
    ----------
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
    """
    def __init__(self,
                 timebin_dur: float,
                 unlabeled_label: int = 0,
                 min_segment_dur: float | None = None,
                 majority_vote: bool = False,
                 ):
        self.timebin_dur = timebin_dur
        self.unlabeled_label = unlabeled_label
        self.min_segment_dur = min_segment_dur
        self.majority_vote = majority_vote

    def __call__(self,
                 lbl_tb: np.ndarray) -> np.ndarray:
        """Convert vector of labeled timebins into labels.

        Parameters
        ----------
        lbl_tb : numpy.ndarray
            Vector of labeled spectrogram time bins, i.e.,
            where each element is a label for a time bin.
            Output of a neural network.

        Returns
        -------
        lbl_tb : numpy.ndarray
            Vector of labeled timebins after post-processing is applied.
        """
        return F.postprocess(lbl_tb, self.timebin_dur, self.unlabeled_label,
                             self.min_segment_dur, self.majority_vote)
