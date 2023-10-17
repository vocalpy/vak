"""Functions for creating splits of datasets used with neural network models,
such as the standard train-val-test splits used with supervised learning methods."""
from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd

from ...common.labels import from_df as labels_from_df
from .algorithms import brute_force
from .algorithms.validate import validate_split_durations

logger = logging.getLogger(__name__)


def train_test_dur_split_inds(
    durs,
    labels,
    labelset,
    train_dur,
    test_dur,
    val_dur=None,
    algo="brute_force",
):
    """Return indices to split a dataset into training, test, and validation sets of specified durations.

    Given the durations of a set of vocalizations, and labels from the annotations for those vocalizations,
    this function returns arrays of indices for splitting up the set into training, test,
    and validation sets.

    Using those indices will produce datasets that each contain instances of all labels in the set of labels.

    Parameters
    ----------
    durs : iterable
        Of float. Durations of audio files.
    labels : iterable
        Of numpy arrays of str or int. Labels for segments (syllables, phonemes, etc.) in audio files.
    labelset : set, list
        set of unique labels for segments in files. Used to verify that each returned array
        of indices will produce a set that contains instances of all labels found in original
        set.
    train_dur : float
        Target duration for training set, in seconds.
    test_dur : float
        Target duration for test set, in seconds.
    val_dur : float
        Target duration for validation set, in seconds. Default is None.
        If None, no indices are returned for validation set.
    algo : str
        algorithm to use. One of {'brute_force', 'inc_freq'}. Default is 'brute_force'. For more information
        on the algorithms, see the docstrings, e.g., vak.io.algorithms.brute_force

    Returns
    -------
    train_inds, test_inds, val_inds : numpy.ndarray
        indices to use with some array-like object to produce sets of specified durations
    """
    if len(durs) != len(labels):
        raise ValueError(
            f"length of durs, {len(durs)} does not equal length of labels, {len(labels)}"
        )

    total_dur = sum(durs)
    train_dur, val_dur, test_dur = validate_split_durations(
        train_dur, val_dur, test_dur, total_dur
    )

    total_target_dur = sum(
        [dur for dur in (train_dur, test_dur, val_dur) if dur is not None]
    )

    if total_target_dur > total_dur:
        raise ValueError(
            f"Total duration of dataset, {total_dur} seconds, is less than total target duration of "
            f"training, test, and (if specified) validation sets: {total_target_dur}"
        )

    logger.info(
        f"Total target duration of splits: {total_target_dur} seconds. "
        f"Will be drawn from dataset with total duration: {total_dur:.3f}.",
    )

    if algo == "brute_force":
        train_inds, val_inds, test_inds = brute_force(
            durs, labels, labelset, train_dur, val_dur, test_dur
        )
    else:
        raise NotImplementedError(f"algorithm {algo} not implemented")

    return train_inds, val_inds, test_inds


def frame_classification_dataframe(
    dataset_df: pd.DataFrame,
    dataset_path: str | pathlib.Path,
    labelset: set,
    train_dur: float | None = None,
    test_dur: float | None = None,
    val_dur: float | None = None,
):
    """Create datasets splits from a dataframe
    representing a frame classification dataset.

    Splits dataset into training, test, and (optionally) validation subsets,
    specified by their duration.

    Additionally, adds a 'split' column to the dataframe,
    that assigns each row to 'train', 'val', 'test', or 'None'.

    Parameters
    ----------
    dataset_df : pandas.Dataframe
        A pandas DataFrame representing the samples in a dataset generated by ``vak prep``.
    dataset_path : str
        Path to dataset, a directory generated by running ``vak prep``.
    labelset : set, list
        The set of label classes for vocalizations in dataset.
    train_dur : float
        Total duration of training set, in seconds. Default is None
    test_dur : float
        Total duration of test set, in seconds. Default is None.
    val_dur : float
        Total duration of validation set, in seconds. Default is None.

    Returns
    -------
    dataset_df : pandas.Dataframe
        A copy of the input dataset with a 'split' column added,
        that assigns each vocalization (row) to a subset,
        i.e., train, validation, or test.
        If the vocalization was not added to one of the subsets,
        its value for 'split' will be 'None'.

    Notes
    -----
    Uses the function :func:`vak.dataset.split.train_test_dur_split_inds`
    to find indices for each subset.
    """
    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    dataset_df = (
        dataset_df.copy()
    )  # don't want this function to have unexpected side effects, so return a copy
    labels = labels_from_df(dataset_df, dataset_path)

    durs = dataset_df["duration"].values
    train_inds, val_inds, test_inds = train_test_dur_split_inds(
        durs=durs,
        labels=labels,
        labelset=labelset,
        train_dur=train_dur,
        test_dur=test_dur,
        val_dur=val_dur,
    )

    # start off with all elements set to 'None'
    # so we don't have to change any that are not assigned to one of the subsets to 'None' after
    split_col = np.asarray(
        ["None" for _ in range(len(dataset_df))], dtype="object"
    )
    split_zip = zip(
        ["train", "val", "test"], [train_inds, val_inds, test_inds]
    )
    for split_name, split_inds in split_zip:
        if split_inds is not None:
            split_col[split_inds] = split_name

    # add split column to dataframe
    dataset_df["split"] = split_col

    return dataset_df


def unit_dataframe(
    dataset_df: pd.DataFrame,
    dataset_path: str | pathlib.Path,
    labelset: set,
    train_dur: float | None = None,
    test_dur: float | None = None,
    val_dur: float | None = None,
):
    """Create datasets splits from a dataframe
    representing a unit dataset.

    Splits dataset into training, test, and (optionally) validation subsets,
    specified by their duration.

    Additionally adds a 'split' column to the dataframe,
    that assigns each row to 'train', 'val', 'test', or 'None'.

    Parameters
    ----------
    dataset_df : pandas.Dataframe
        A pandas DataFrame representing the samples in a dataset,
        generated by ``vak prep``.
    dataset_path : str
        Path to dataset, a directory generated by running ``vak prep``.
    labelset : set, list
        The set of label classes for vocalizations in dataset.
    train_dur : float
        Total duration of training set, in seconds. Default is None
    test_dur : float
        Total duration of test set, in seconds. Default is None.
    val_dur : float
        Total duration of validation set, in seconds. Default is None.

    Returns
    -------
    dataset_df : pandas.Dataframe
        A copy of the input dataset with a 'split' column added,
        that assigns each vocalization (row) to a subset,
        i.e., train, validation, or test.
        If the vocalization was not added to one of the subsets,
        its value for 'split' will be 'None'.

    Notes
    -----
    Uses the function :func:`vak.dataset.split.train_test_dur_split_inds`
    to find indices for each subset.
    """
    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    dataset_df = (
        dataset_df.copy()
    )  # don't want this function to have unexpected side effects, so return a copy
    labels = [np.array([label]) for label in dataset_df.label.values]

    durs = dataset_df["duration"].values
    train_inds, val_inds, test_inds = train_test_dur_split_inds(
        durs=durs,
        labels=labels,
        labelset=labelset,
        train_dur=train_dur,
        test_dur=test_dur,
        val_dur=val_dur,
    )

    # start off with all elements set to 'None'
    # so we don't have to change any that are not assigned to one of the subsets to 'None' after
    split_col = np.asarray(
        ["None" for _ in range(len(dataset_df))], dtype="object"
    )
    split_zip = zip(
        ["train", "val", "test"], [train_inds, val_inds, test_inds]
    )
    for split_name, split_inds in split_zip:
        if split_inds is not None:
            split_col[split_inds] = split_name

    # add split column to dataframe
    dataset_df["split"] = split_col

    return dataset_df
