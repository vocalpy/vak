from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from . import annotation, constants


def to_map(
    labelset: set,
    map_background: bool = True,
    background_label: str = constants.DEFAULT_BACKGROUND_LABEL,
) -> dict:
    """Convert set of labels to `dict`
    mapping those labels to a series of consecutive integers
    from 0 to n inclusive,
    where n is the number of labels in the set.

    This 'labelmap' is used when mapping labels
    from annotations of a vocalization into
    a label for every time bin in a spectrogram of that vocalization.

    If ``map_background`` is True, then a label
    will be added to labelset representing a background class
    (any segment that is not labeled).
    The default for this label is
    :const:`vak.common.constants.DEFAULT_BACKGROUND_LABEL`.
    This string label will map to class index 0,
    so the total number of classes is n + 1.

    Parameters
    ----------
    labelset : set
        Set of labels used to annotate a dataset.
    map_background : bool
        If True, include key specified by
        ``background_label`` in mapping.
        Any time bins in a spectrogram
        that do not have a label associated with them,
        e.g. a silent gap between vocalizations,
        will be assigned the integer
        that the background key maps to.
    background_label: str, optional
        The string label applied to segments belonging to the
        background class.
        Default is
        :const:`vak.common.constants.DEFAULT_BACKGROUND_LABEL`.

    Returns
    -------
    labelmap : dict
        Maps labels to integers.
    """
    if not isinstance(labelset, set):
        raise TypeError(
            f"type of labelset must be set, got type {type(labelset)}"
        )

    labellist = []
    if map_background is True:
        # NOTE we append background label *first*
        labellist.append(background_label)
    # **then** extend with the rest of the labels
    labellist.extend(sorted(list(labelset)))
    # so that background_label maps to class index 0 by default in next line
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


def from_df(
    dataset_df: pd.DataFrame, dataset_path: str | pathlib.Path
) -> list[np.ndarray]:
    """Returns labels for each vocalization in a dataset.

    Takes Pandas DataFrame representing the dataset, loads
    annotation for each row in the DataFrame, and then returns
    labels from each annotation.

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        created by vak.io.dataframe.from_files

    Returns
    -------
    labels : list
        of array-like, labels for each vocalization in the dataset.
    """
    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    annots = annotation.from_df(dataset_df, dataset_path)
    return [annot.seq.labels for annot in annots]


ALPHANUMERIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
DUMMY_SINGLE_CHAR_LABELS = [
    chr(x)
    # some large range of characters not typically used as labels
    for x in range(162, 2000)
]
# start with alphanumeric since more human readable;
# mapping can be arbitrary as long as it's consistent
DUMMY_SINGLE_CHAR_LABELS = (*ALPHANUMERIC, *DUMMY_SINGLE_CHAR_LABELS)


# added to fix https://github.com/NickleDave/vak/issues/373
def multi_char_labels_to_single_char(
    labelmap: dict, skip: tuple[str] = (constants.DEFAULT_BACKGROUND_LABEL,)
) -> dict:
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
        A tuple of labels to leave as multiple characters.
        Default is a tuple containing just
        :const:`vak.common.constants.DEFAULT_BACKGROUND_LABEL`.

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
    if all([len(lbl) == 1 for lbl in current_str_labels]):
        # no need to do re-mapping
        return labelmap

    # We only use single character labels that are not already in labelmap,
    # to avoid over-writing a single-character label from the original labelmap
    # with the same single-character from DUMMY_SINGLE_CHAR_LABELS,
    # which would map it to a new integer and cause us to lose the original integer
    # from the mapping
    single_char_labels_not_in_labelmap = [
        lbl for lbl in DUMMY_SINGLE_CHAR_LABELS if lbl not in labelmap
    ]
    n_needed_to_remap = len(
        [lbl for lbl in current_str_labels if len(lbl) > 1]
    )
    if n_needed_to_remap > len(single_char_labels_not_in_labelmap):
        raise ValueError(
            f"Need to remap {n_needed_to_remap} multiple-character labels"
            f"but there are only {len(single_char_labels_not_in_labelmap)} available."
        )

    new_labelmap = {}
    for dummy_label_ind, label_str in enumerate(current_str_labels):
        label_int = labelmap[label_str]
        if (
            len(label_str) > 1 and label_str not in skip
        ):  # default for `skip` is ('unlabeled',)
            # replace with dummy label
            new_label_str = single_char_labels_not_in_labelmap[dummy_label_ind]
            new_labelmap[new_label_str] = label_int
        else:
            new_labelmap[label_str] = label_int
    return new_labelmap
