import numpy as np

from .validation import column_or_1d


def label_timebins(labels,
                   onsets,
                   offsets,
                   time_bins,
                   silent_gap_label=0):
    """makes a vector of labels for each timebin from a spectrogram,
    given labels for syllables plus onsets and offsets of syllables

    Parameters
    ----------
    labels : ints
        should be mapping returned by make_labels_mapping
    onsets : ndarray
        1d vector of floats, syllable onsets in seconds
    offsets : ndarray
        1d vector of floats, offsets in seconds
    time_bins : ndarray
        1d vector of floats,
        time in seconds for each time bin of a spectrogram
    silent_gap_label : int
        label assigned to silent gaps
        default is 0

    Returns
    -------
    label_vec : ndarray
        same length as time_bins, with each element a label for
        each time bin
    """
    labels = [int(label) for label in labels]
    label_vec = np.ones((time_bins.shape[-1], 1), dtype='int8') * silent_gap_label
    onset_inds = [np.argmin(np.abs(time_bins - onset))
                  for onset in onsets]
    offset_inds = [np.argmin(np.abs(time_bins - offset))
                   for offset in offsets]
    for label, onset, offset in zip(labels, onset_inds, offset_inds):
        label_vec[onset:offset+1] = label
        # offset_inds[ind]+1 because of Matlab one-indexing
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
