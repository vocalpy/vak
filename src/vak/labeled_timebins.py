"""functions for dealing with labeled timebin vectors"""
import numpy as np
import scipy.stats

from .timebins import timebin_dur_from_vec
from .validators import row_or_1d, column_or_1d


def has_unlabeled(labels_int,
                  onsets_s,
                  offsets_s,
                  time_bins):
    """determine whether there are unlabeled segments in a spectrogram,
    given labels, onsets, and offsets of vocalizations, and vector of
    time bins from spectrogram

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
    if (type(labels_int) == list and not all([type(lbl) == int for lbl in labels_int]) or
            (type(labels_int) == np.ndarray and labels_int.dtype not in [np.int8, np.int16, np.int32, np.int64])):
        raise TypeError('labels_int must be a list or numpy.ndarray of integers')

    dummy_unlabeled_label = np.max(labels_int) + 1
    label_vec = np.ones((time_bins.shape[-1], 1), dtype='int8') * dummy_unlabeled_label
    onset_inds = [np.argmin(np.abs(time_bins - onset))
                  for onset in onsets_s]
    offset_inds = [np.argmin(np.abs(time_bins - offset))
                   for offset in offsets_s]
    for label, onset, offset in zip(labels_int, onset_inds, offset_inds):
        # offset_inds[ind]+1 because offset time bin is still "part of" syllable
        label_vec[onset:offset+1] = label

    if dummy_unlabeled_label in label_vec:
        return True
    else:
        return False


def label_timebins(labels_int,
                   onsets_s,
                   offsets_s,
                   time_bins,
                   unlabeled_label=0):
    """makes a vector of labels for each time bin from a spectrogram,
    given labels, onsets, and offsets of vocalizations

    Parameters
    ----------
    labels_int : list, numpy.ndarray
        a list or array of labels from the annotation for a vocalization,
        mapped to integers
    onsets_s : numpy.ndarray
        1d vector of floats, segment onsets in seconds
    offsets_s : numpy.ndarray
        1-d vector of floats, segment offsets in seconds
    time_bins : mumpy.ndarray
        1-d vector of floats, time in seconds for center of each time bin of a spectrogram
    unlabeled_label : int
        label assigned to time bins that do not have labels associated with them.
        Default is 0

    Returns
    -------
    lbl_tb : numpy.ndarray
        same length as time_bins, with each element a label for each time bin
    """
    if (type(labels_int) == list and not all([type(lbl) == int for lbl in labels_int]) or
            (type(labels_int) == np.ndarray and labels_int.dtype not in [np.int8, np.int16, np.int32, np.int64])):
        raise TypeError('labels_int must be a list or numpy.ndarray of integers')

    label_vec = np.ones((time_bins.shape[-1],), dtype='int8') * unlabeled_label
    onset_inds = [np.argmin(np.abs(time_bins - onset))
                  for onset in onsets_s]
    offset_inds = [np.argmin(np.abs(time_bins - offset))
                   for offset in offsets_s]
    for label, onset, offset in zip(labels_int, onset_inds, offset_inds):
        # offset_inds[ind]+1 because offset time bin is still "part of" syllable
        label_vec[onset:offset+1] = label

    return label_vec


def lbl_tb2labels(labeled_timebins,
                  labels_mapping,
                  spect_ID_vector=None):
    """converts output of network from label for each frame
    to one label for each continuous segment

    Parameters
    ----------
    labeled_timebins : ndarray
        where each element is a label for a time bin.
        Such an array is the output of the network.
    labels_mapping : dict
        that maps str labels to consecutive integers.
        The mapping is inverted to convert back to str labels.
    spect_ID_vector : ndarray
        of same length as labeled_timebins, where each element
        is an ID # for the spectrogram from which labeled_timebins
        was taken.
        If provided, used to split the converted labels back to
        a list of label str, with one for each spectrogram.
        Default is None, in which case the return value is one long str.

    Returns
    -------
    labels : str or list
        labeled_timebins mapped back to label str.
        If spect_ID_vector was provided, then labels is split into a list of str,
        where each str corresponds to predicted labels for each predicted
        segment in each spectrogram as identified by spect_ID_vector.
    """
    labeled_timebins = row_or_1d(labeled_timebins)
    idx = np.diff(labeled_timebins, axis=0).astype(np.bool)
    idx = np.insert(idx, 0, True)

    labels = labeled_timebins[idx]

    # remove 'unlabeled' label
    if 'unlabeled' in labels_mapping:
        labels = labels[labels != labels_mapping['unlabeled']]

    inverse_labels_mapping = dict((v, k) for k, v
                                  in labels_mapping.items())
    labels = labels.tolist()
    labels = [inverse_labels_mapping[label] for label in labels]

    if spect_ID_vector:
        labels_list = []
        spect_ID_vector = spect_ID_vector[idx]
        labels_arr = np.asarray(labels)
        # need to split up labels by spect_ID_vector
        # this is probably not the most efficient way:
        spect_IDs = np.unique(spect_ID_vector)

        for spect_ID in spect_IDs:
            these = np.where(spect_ID_vector == spect_ID)
            curr_labels = labels_arr[these].tolist()
            if all([type(el) is str for el in curr_labels]):
                labels_list.append(''.join(curr_labels))
            elif all([type(el) is int for el in curr_labels]):
                labels_list.append(curr_labels)
        return labels_list, spect_ID_vector
    else:
        if all([type(el) is str or type(el) is np.str_ for el in labels]):
            return ''.join(labels)
        elif all([type(el) is int for el in labels]):
            return labels


def _segment_lbl_tb(lbl_tb):
    """helper function that segments vector of labeled timebins.

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        vector where each element represents a label for a timebin

    Returns
    -------
    labels : numpy.ndarray
        vector where each element is a label for a segment with its onset
        and offset indices given by the corresponding element in onset_inds
        and offset_inds.
    onset_inds : numpy.ndarray
        vector where each element is the onset index for a segment.
        Each onset corresponds to the value at the same index in labels.
    offset_inds : numpy.ndarray
        vector where each element is the offset index for a segment
        Each offset corresponds to the value at the same index in labels.
    """
    # factored out as a separate function to be able to test
    # and in case user wants to do just this with output of neural net
    offset_inds = np.where(np.diff(lbl_tb, axis=0))[0]
    onset_inds = offset_inds + 1
    offset_inds = np.concatenate(
        (offset_inds, np.asarray([lbl_tb.shape[0] - 1]))
    )
    onset_inds = np.concatenate(
        (np.asarray([0]), onset_inds)
    )
    labels = lbl_tb[onset_inds]
    return labels, onset_inds, offset_inds


def lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=0):
    """given a vector of labeled timebins,
    returns a list of indexing vectors,
    one for each labeled segment in the vector.

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        vector of labeled timebins from spectrogram
    unlabeled_label : int
        label that was given to segments that were not labeled in annotation,
        e.g. silent periods between annotated segments. Default is 0.
    return_inds : bool
        if True, return list of indices for segments in lbl_tb, in addition to the segments themselves.
        if False, just return list of numpy.ndarrays that are the segments from lbl_tb.

    Returns
    -------
    segment_inds_list : list
        of numpy.ndarray, indices that will recover segments list from lbl_tb.
    """
    segment_inds = np.nonzero(lbl_tb != unlabeled_label)[0]
    return np.split(segment_inds, np.where(np.diff(segment_inds) != 1)[0] + 1)


def remove_short_segments(lbl_tb,
                          segment_inds_list,
                          timebin_dur,
                          min_segment_dur,
                          unlabeled_label=0):
    """remove segments from vector of labeled timebins
    that are shorter than specified duration

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        vector of labeled spectrogram time bins, i.e.,
        where each element is a label for a time bin.
        Output of a neural network.
    segment_inds_list : list
        of numpy.ndarray, indices that will recover segments list from lbl_tb.
        Returned by funciton ``vak.labels.lbl_tb_segment_inds_list``.
    timebin_dur : float
        Duration of a single timebin in the spectrogram, in seconds.
        Used to convert onset and offset indices in lbl_tb to seconds.
    min_segment_dur : float
        minimum duration of segment, in seconds. If specified, then
        any segment with a duration less than min_segment_dur is
        removed from lbl_tb. Default is None, in which case no
        segments are removed.
    unlabeled_label : int
        label that was given to segments that were not labeled in annotation,
        e.g. silent periods between annotated segments. Default is 0.

    Returns
    -------
    lbl_tb : numpy.ndarray
        with segments removed whose duration is shorter than min_segment_dur
    segment_inds_list : list
        of numpy.ndarray, with arrays popped off that correspond
        to segments removed from lbl_tb
    """
    to_pop = []

    for segment_inds in segment_inds_list:
        if segment_inds.shape[-1] * timebin_dur < min_segment_dur:
            lbl_tb[segment_inds] = unlabeled_label

    if to_pop:
        for seg_inds in to_pop:
            segment_inds_list.remove(seg_inds)

    return lbl_tb, segment_inds_list


def majority_vote_transform(lbl_tb,
                            segment_inds_list):
    """transform segments containing multiple labels
        into segments with a single label by taking a "majority vote",
        i.e. assign all time bins in the segment the most frequently
        occurring label in the segment.

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        vector of labeled spectrogram time bins, i.e.,
        where each element is a label for a time bin.
        Output of a neural network.
    segment_inds_list : list
        of numpy.ndarray, indices that will recover segments list from lbl_tb.
        Returned by funciton ``vak.labels.lbl_tb_segment_inds_list``.

    Returns
    -------
    lbl_tb : numpy.ndarray
        after the majority vote transform has been applied
    """
    for segment_inds in segment_inds_list:
        segment = lbl_tb[segment_inds]
        majority = scipy.stats.mode(segment)[0].item()
        lbl_tb[segment_inds] = majority

    return lbl_tb


def lbl_tb2segments(lbl_tb,
                    labelmap,
                    t,
                    min_segment_dur=None,
                    majority_vote=False,
                    n_decimals_trunc=5):
    """convert vector of labeled timebins into segments,
    by finding where continuous runs of a single label start
    and stop. Returns vectors of labels and onsets and offsets
    in units of seconds.

    Parameters
    ----------
    lbl_tb : numpy.ndarray
        vector of labeled spectrogram time bins, i.e.,
        where each element is a label for a time bin.
        Output of a neural network.
    labelmap : dict
        that maps labels to consecutive integers.
        The mapping is inverted to convert back to labels.
    t : numpy.ndarray
        Vector of times; the times are bin centers of columns in a spectrogram.
        Returned by function that generated spectrogram.
        Used to convert onset and offset indices in lbl_tb to seconds.
    min_segment_dur : float
        minimum duration of segment, in seconds. If specified, then
        any segment with a duration less than min_segment_dur is
        removed from lbl_tb. Default is None, in which case no
        segments are removed.
    majority_vote : bool
        if True, transform segments containing multiple labels
        into segments with a single label by taking a "majority vote",
        i.e. assign all time bins in the segment the most frequently
        occurring label in the segment. This transform can only be
        applied if the labelmap contains an 'unlabeled' label,
        because unlabeled segments makes it possible to identify
        the labeled segments. Default is False.
    n_decimals_trunc : int
        number of decimal places to keep when truncating the timebin duration
        calculated from the vector of times t. Default is 5.

    Returns
    -------
    labels : numpy.ndarray
        vector where each element is a label for a segment with its onset
        and offset indices given by the corresponding element in onset_inds
        and offset_inds.
    onsets_s : numpy.ndarray
        vector where each element is the onset in seconds a segment.
        Each onset corresponds to the value at the same index in labels.
    offsets_s : numpy.ndarray
        vector where each element is the offset in seconds of a segment.
        Each offset corresponds to the value at the same index in labels.
    """
    lbl_tb = column_or_1d(lbl_tb)

    timebin_dur = timebin_dur_from_vec(t, n_decimals_trunc)

    if min_segment_dur is not None or majority_vote:
        if 'unlabeled' not in labelmap:
            raise ValueError(
                "min_segment_dur or majority_vote specified,"
                " but 'unlabeled' not in labelmap.\n"
                "Without 'unlabeled' segments these transforms cannot be applied."
            )
        segment_inds_list = lbl_tb_segment_inds_list(lbl_tb,
                                                     unlabeled_label=labelmap['unlabeled'])

    if min_segment_dur is not None:
        lbl_tb, segment_inds_list = remove_short_segments(lbl_tb,
                                                          segment_inds_list,
                                                          timebin_dur,
                                                          min_segment_dur,
                                                          labelmap['unlabeled'],
                                                          )

    if majority_vote:
        lbl_tb = majority_vote_transform(lbl_tb, segment_inds_list)

    labels, onset_inds, offset_inds = _segment_lbl_tb(lbl_tb)

    # remove 'unlabeled' label
    if 'unlabeled' in labelmap:
        keep = np.where(labels != labelmap['unlabeled'])[0]
        labels = labels[keep]
        onset_inds = onset_inds[keep]
        offset_inds = offset_inds[keep]
    inverse_labelmap = dict((v, k) for k, v
                            in labelmap.items())
    labels = labels.tolist()
    labels = np.asarray(
        [inverse_labelmap[label] for label in labels]
    )
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

    return labels, onsets_s, offsets_s
