from . import annotation


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
