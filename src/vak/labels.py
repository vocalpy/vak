from __future__ import annotations

import numpy as np
import pandas as pd

from . import annotation


def to_map(labelset: set,
           map_unlabeled: bool = True) -> dict:
    """Convert set of labels to `dict`
    mapping those labels to a series of consecutive integers
    from 0 to n inclusive,
    where n is the number of labels in the set.

    This 'labelmap' is used when mapping labels
    from annotations of a vocalization into
    a label for every time bin in a spectrogram of that vocalization.

    If ``map_unlabeled`` is True, then the label 'unlabeled'
    will be added to labelset, and will map to 0,
    so the total number of classes is n + 1.

    Parameters
    ----------
    labelset : set
        Set of labels used to annotate a dataset.
    map_unlabeled : bool
        If True, include key 'unlabeled' in mapping.
        Any time bins in a spectrogram
        that do not have a label associated with them,
        e.g. a silent gap between vocalizations,
        will be assigned the integer
        that the 'unlabeled' key maps to.

    Returns
    -------
    labelmap : dict
        Maps labels to integers.
    """
    if type(labelset) != set:
        raise TypeError(f"type of labelset must be set, got type {type(labelset)}")

    labellist = []
    if map_unlabeled is True:
        labellist.append("unlabeled")

    labellist.extend(sorted(list(labelset)))

    labelmap = dict(zip(labellist, range(len(labellist))))
    return labelmap


def to_set(labels_list: list[np.ndarray | list]) -> set:
    """Given a list of labels from annotations,
    return the set of (unique) labels.

    Parameters
    ----------
    labels_list : list
         Of labels from annotations,
         either a list of numpy.ndarrays
         or a list of lists.

    Returns
    -------
    labelset : set
        Unique set of labels found in ``labels_list``.

    Examples
    --------
    >>> labels_list = [voc.annot.labels for voc in vds.voc_list]
    >>> labelset = to_set(labels_list)
    >>> print(labelset)
    {'a', 'b', 'c', 'd', 'e'}
    """
    all_labels = [lbl for labels in labels_list for lbl in labels]
    labelset = set(all_labels)
    return labelset


def from_df(vak_df: pd.DataFrame) -> list[np.ndarray]:
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


ALPHANUMERIC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
DUMMY_SINGLE_CHAR_LABELS = [
    # some large range of characters not typically used as labels
    chr(x) for x in range(162, 400)
]
# start with alphanumeric since more human readable;
# mapping can be arbitrary as long as it's consistent
DUMMY_SINGLE_CHAR_LABELS = (
    *ALPHANUMERIC,
    *DUMMY_SINGLE_CHAR_LABELS
)


# added to fix https://github.com/NickleDave/vak/issues/373
def multi_char_labels_to_single_char(labelmap: dict, skip: tuple[str] = ('unlabeled',)) -> dict:
    """Return a copy of a ``labelmap`` where any
    labels that are strings with multiple characters
    are converted to single characters.

    This makes it possible to correctly compute metrics
    like Levenshtein edit distance.

    Labels that are strings with multiple characters
    are replaced by a single-label character from
    the constant ``vak.labels.DUMMY_SINGLE_CHAR_LABELS``.
    The replacement is grabbed with the index of the
    multi-character label from the sorted ``dict``.

    Parameters
    ----------
    labelmap : dict
        That maps human-readable string labels
        to integers. As returned by
        ``vak.labels.to_map``.
    skip : tuple
        Of strings, labels to leave
        as multiple characters.
        Default is ('unlabeled',).

    Returns
    -------
    labelmap : dict
        Where any keys with multiple characters
        in string are converted to dummy single characters.
    """
    current_str_labels = sorted(
        # sort to be extra sure we get same order every time
        # (even though OrderedDict is now default in Python).
        # Same order forces mapping to single characters to be deterministic across function calls.
        labelmap.keys()
    )
    new_labelmap = {}
    for dummy_label_ind, label_str in enumerate(current_str_labels):
        label_int = labelmap[label_str]
        if len(label_str) > 1 and label_str not in skip:
            # replace with dummy label
            new_label_str = DUMMY_SINGLE_CHAR_LABELS[dummy_label_ind]
            new_labelmap[new_label_str] = label_int
        else:
            new_labelmap[label_str] = label_int
    return new_labelmap
