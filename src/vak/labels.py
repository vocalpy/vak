import numpy as np
import scipy.stats

from . import annotation
from .validation import column_or_1d


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


def to_map(labelset, map_unlabeled=True):
    """map set of labels to series of consecutive integers from 0 to n inclusive,
    where n is the number of labels in the set.

    This 'labelmap' is used when mapping labels from annotations of a vocalization into
    a label for every time bin in a spectrogram of that vocalization.

    If map_unlabeled is True, 'unlabeled' will be added to labelset, and will map to 0,
    so the total number of classes is n + 1.

    Parameters
    ----------
    labelset : set
        of labels used to annotate a Dataset.
    map_unlabeled : bool
        if True, include key 'unlabeled' in mapping. Any time bins in a spectrogram
        that do not have a label associated with them, e.g. a silent gap between vocalizations,
        will be assigned the integer that the 'unlabeled' key maps to.

    Returns
    -------
    labelmap : dict
        maps labels to integers
    """
    if type(labelset) != set:
        raise TypeError(
            f'type of labelset must be set, got type {type(labelset)}'
        )

    labellist = []
    if map_unlabeled is True:
        labellist.append('unlabeled')

    labellist.extend(
        sorted(list(labelset))
    )

    labelmap = dict(
        zip(
            labellist, range(len(labellist))
        )
    )
    return labelmap


def to_set(labels_list):
    """given a list of labels from annotations, return the set of (unique) labels

    Parameters
    ----------
    labels_list : list
         of lists, i.e. labels from annotations

    Returns
    -------
    labelset

    Examples
    --------
    >>> labels_list = [voc.annot.labels for voc in vds.voc_list]
    >>> labelset = to_set(labels_list)
    """
    all_labels = [lbl for labels in labels_list for lbl in labels]
    labelset = set(all_labels)
    return labelset


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
    idx = np.diff(labeled_timebins, axis=0).astype(np.bool)
    idx = np.insert(idx, 0, True)

    labels = labeled_timebins[idx]

    # remove 'unlabeled' label
    if 'unlabeled' in labels_mapping:
        labels = labels[labels != labels_mapping['unlabeled']]
        labels = labels.tolist()

    inverse_labels_mapping = dict((v, k) for k, v
                                  in labels_mapping.items())
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


def _contiguous(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.

    From:
    https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array/4495197#4495197
    """
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


class MajorityVote:
    """transform that converts a segment with multiple labels
    into a segment with a single label, whichever one is in
    the majority

    finds contiguous labeled segments
    within a vector of labeled timebins.
    Transform only works if vector also has
    unlabeled segments.

    Attributes
    ----------
    unlabeled : int
        integer value that represents unlabeled segments
        within a vector of labeled timebins. Default is 0.
    """
    def __init__(self, unlabeled=0):
        unlabeled = int(unlabeled)
        self.unlabeled = unlabeled

    def __call__(self, lbl_tb):
        condition = lbl_tb != self.unlabeled
        idx = _contiguous(condition)
        for row in idx:
            onset, offset = row[0], row[1]
            lbl_tb[onset:offset] = scipy.stats.mode(lbl_tb[onset:offset])[0].item()
        return lbl_tb


def lbl_tb2segments(lbl_tb,
                    labelmap,
                    timebin_dur,
                    majority_vote=False):
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
    timebin_dur : float
        Duration of a single timebin in the spectrogram, in seconds.
        Used to convert onset and offset indices in lbl_tb to seconds.
    majority_vote : bool
        if True, transform segments containing multiple labels
        into segments with a single label by taking a "majority vote",
        i.e. assign all time bins in the segment the most frequently
        occurring label in the segment. This transform can only be
        applied if the labelmap contains an 'unlabeled' label,
        because unlabeled segments makes it possible to identify
        the labeled segments. Default is False.

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

    if majority_vote:
        if 'unlabeled' in labelmap:
            transform = MajorityVote(unlabeled=labelmap['unlabeled'])
            lbl_tb = transform(lbl_tb)
        else:
            raise ValueError(
                "majority_vote set to True, but labelmap does not contain 'unlabeled'; "
                "unclear how to determine onset and offset of segments for which the "
                "labels should be converted to whichever is in the majority"
            )

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
    onsets_s = onset_inds * timebin_dur
    offsets_s = offset_inds * timebin_dur

    return labels, onsets_s, offsets_s


def from_df(vak_df):
    """returns labels for each vocalization in a dataset.
    Takes Pandas DataFrame representing the dataset, loads
    annotation for each row in the DataFrame, and then returns
    labels from each annotation.

    Parameters
    ----------
    vak_df : pandas.DataFrame
        created by vak.io.dataframe.from_files

    Returns
    -------
    labels : list
        of array-like, labels for each vocalization in the dataset.
    """
    annots = annotation.from_df(vak_df)
    return [annot.seq.labels for annot in annots]
