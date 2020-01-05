import crowsetta
import numpy as np

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


def translate(label_arr, map_dict):
    return np.vectorize(map_dict.__getitem__)(label_arr)


def where(label_arrs, find_in_arr=False):
    """given a list of label arrays for a set of audio files,
    return for each unique label the indices of the arrays in which it occurs
    and optionally where it occurs in each array of labels.

    What is actually returned are index arrays for each unique label.
    These can be used for example to index into a list of file names that corresponds
    to the list of label arrays passed to this function.

    Parameters
    ----------
    label_arrs : list, tuple
        of arrays of labels for segments in a set of audio files, e.g. an array
        of labels for animal vocalizations and silent periods within each file.
        Each element in the iterable should correspond to a labels array for one file.
    find_in_arr : bool
        if True, find where each unique label occurs in each array of labels, i.e.,
        also return indices of each occurrence within all arrays.
        Default is False.

    Returns
    -------
    where_in_label_arrs : dict
        in which keys are unique labels, and the corresponding
        value for each key is an array of indices that would select
        all elements in labels that
    where_in_each_arr : dict
        in which keys are unique labels, and the corresponding
        value for each key is a dictionary where the keys are indices
        into label_arrs, and the values are index arrays that would
        select just the specified label from the array it is mapped to.
    """
    if type(label_arrs) not in [list, tuple]:
        raise TypeError(f"label_arrs must be a list or tuple, but was: {type(labels)}")

    label_arrs = [column_or_1d(lbl) for lbl in label_arrs]

    label_arrs_inds = []
    all_lbl_arr = []

    for ind, label_arr in enumerate(label_arrs):
        all_lbl_arr.append(label_arr)
        label_arrs_inds.append(
            np.ones(shape=label_arr.shape[0], dtype=np.int64) * ind
        )

    all_lbl_arr = np.concatenate(all_lbl_arr)
    label_arrs_inds = np.concatenate(label_arrs_inds)

    uniq_lbl = np.unique(all_lbl_arr)
    where_in_labels = {}

    for this_lbl in uniq_lbl:
        in_all_lbl = np.where(all_lbl_arr == this_lbl)[0]
        this_lbl_label_arrs_inds = label_arrs_inds[in_all_lbl]
        uniq_label_arrs_inds = np.unique(this_lbl_label_arrs_inds)
        where_in_labels[this_lbl] = uniq_label_arrs_inds

    if find_in_arr:
        where_in_arr = {}
        for this_lbl, these_inds in where_in_labels.items():
            ind_arr_map = {}
            for this_ind in these_inds:
                in_this_arr = np.where(label_arrs[this_ind] == this_lbl)[0]
                ind_arr_map[this_ind] = in_this_arr
            where_in_arr[this_lbl] = ind_arr_map

    if find_in_arr:
        return where_in_labels, where_in_arr
    else:
        return where_in_labels


def sort(uniq_labels, occurrences, ascending=True):
    """sort a list of unique labels by the number of times they occur.

    Parameters
    ----------
    uniq_labels : list
        list of unique label classes
    occurrences : list
        of index arrays, same length as uniq_labels.
        Each array in the list contains indexes where the corresponding label
        in uniq_labels occurs, e.g. in a set of annotations for audio files, or
        in an array of labels for an individual audio file.
    ascending : bool
        if True, sort in ascending order. if False, return in descending order.
        Default is True.

    Returns
    -------
    uniq_labels_sorted, occurrences_sorted : list
    """
    if len(set(uniq_labels)) != len(uniq_labels):
        raise ValueError('uniq_labels should be a unique set, but found repeated elements')

    occurrences = [column_or_1d(occur_arr) for occur_arr in occurrences]

    counts = [arr.shape[0] for arr in occurrences]
    sort_inds = np.argsort(counts)
    # make lists since occurrences wil have different lengths for each array
    uniq_labels_sorted = []
    occurrences_sorted = []
    for sort_ind in sort_inds:
        uniq_labels_sorted.append(uniq_labels[sort_ind])
        occurrences_sorted.append(occurrences[sort_ind])
    if ascending:
        return uniq_labels_sorted, occurrences_sorted
    else:
        return uniq_labels_sorted.reverse(), occurrences_sorted.reverse()


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


def lbl_tb2segments(lbl_tb,
                    labelmap,
                    timebin_dur):
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
