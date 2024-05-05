"""Assign samples in a dataset to splits.

Given a set of source files represented by a dataframe,
assign each sample (row) to a split.

Helper function called by :func:`vak.prep.frame_classification.prep_frame_classification_dataset`.
"""

from __future__ import annotations

import logging
import pathlib

import pandas as pd

from .. import dataset_df_helper, split

logger = logging.getLogger(__name__)


def assign_samples_to_splits(
    purpose: str,
    dataset_df: pd.DataFrame,
    dataset_path: str | pathlib.Path,
    train_dur: float | None = None,
    val_dur: float | None = None,
    test_dur: float | None = None,
    labelset: set | None = None,
) -> pd.DataFrame:
    """Assign samples in a dataset to splits.

    Given a set of source files represented by a dataframe,
    assign each sample (row) to a split.

    Helper function called by :func:`vak.prep.frame_classification.prep_frame_classification_dataset`.

    If no durations are specified for splits,
    or the purpose is either `'eval'` or `'predict'`,
    then all rows in the dataframe
    will be assigned to ``purpose``.

    Parameters
    ----------
    purpose : str
        Purpose of the dataset.
        One of {'train', 'eval', 'predict', 'learncurve'}.
        These correspond to commands of the vak command-line interface.
    train_dur : float
        Total duration of training set, in seconds.
        When creating a learning curve,
        training subsets of shorter duration
        will be drawn from this set. Default is None.
    val_dur : float
        Total duration of validation set, in seconds.
        Default is None.
    test_dur : float
        Total duration of test set, in seconds.
        Default is None.
    dataset_df : pandas.DataFrame
        That represents a dataset.
    dataset_path : pathlib.Path
        Path to csv saved from ``dataset_df``.
    labelset : str, list, set
        Set of unique labels for vocalizations. Strings or integers.
        Default is ``None``. If not ``None``, then files will be skipped
        where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using
        :func:`vak.converters.labelset_to_set`.
        See help for that function for details on how to specify ``labelset``.

    Returns
    -------
    dataset_df : pandas.DataFrame
        The same ``dataset_df`` with a `'split'` column added,
        where each element in that column assigns the corresponding
        row to one of the splits in the dataset.
    """

    # ---- (possibly) split into train / val / test sets ---------------------------------------------
    # catch case where user specified duration for just training set, raise a helpful error instead of failing silently
    if (purpose == "train" or purpose == "learncurve") and (
        (train_dur is not None and train_dur > 0)
        and (val_dur is None or val_dur == 0)
        and (test_dur is None or val_dur == 0)
    ):
        raise ValueError(
            "A duration specified for just training set, but prep function does not currently support creating a "
            "single split of a specified duration. Either remove the train_dur option from the prep section and "
            "rerun, in which case all data will be included in the training set, or specify values greater than "
            "zero for test_dur (and val_dur, if a validation set will be used)"
        )

    if all(
        [dur is None for dur in (train_dur, val_dur, test_dur)]
    ) or purpose in (
        "eval",
        "predict",
    ):
        # then we're not going to split
        logger.info("Will not split dataset.")
        do_split = False
    else:
        if val_dur is not None and train_dur is None and test_dur is None:
            raise ValueError(
                "cannot specify only val_dur, unclear how to split dataset into training and test sets"
            )
        else:
            logger.info("Will split dataset.")
            do_split = True

    if do_split:
        dataset_df = split.frame_classification_dataframe(
            dataset_df,
            dataset_path,
            labelset=labelset,
            train_dur=train_dur,
            val_dur=val_dur,
            test_dur=test_dur,
        )

    elif (
        do_split is False
    ):  # add a split column, but assign everything to the same 'split'
        # ideally we would just say split=purpose in call to add_split_col, but
        # we have to special case, because "eval" looks for a 'test' split (not an "eval" split)
        if purpose == "eval":
            split_name = (
                "test"  # 'split_name' to avoid name clash with split package
            )
        elif purpose == "predict":
            split_name = "predict"

        dataset_df = dataset_df_helper.add_split_col(
            dataset_df, split=split_name
        )

    return dataset_df
