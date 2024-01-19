"""Functionality to prepare subsets of the 'train' split of segment datasets,
for generating a learning curve."""
from __future__ import annotations

import logging
import pathlib
from typing import Sequence

import pandas as pd

from ... import common
from .. import split

logger = logging.getLogger(__name__)


def make_subsets_from_dataset_df(
    dataset_df: pd.DataFrame,
    train_set_durs: Sequence[float],
    num_replicates: int,
    dataset_path: pathlib.Path,
    labelmap: dict,
) -> pd.DataFrame:
    """Make subsets of the training data split for a learning curve.

     Makes subsets given a dataframe representing the entire dataset,
     with one subset for each combination of (training set duration,
     replicate number). Each subset is randomly drawn
     from the total training split.

     Uses :func:`vak.prep.split.segment_dataset` to make
     subsets of the training data from ``dataset_df``.

     A new column will be added to the dataframe, `'subset'`,
     and additional rows for each subset.
     The dataframe is returned with these subsets added.
     (The `'split'` for these rows will still be `'train'`.)

     Parameters
     ----------
     dataset_df : pandas.DataFrame
         Dataframe representing a dataset for frame classification models.
         It is returned by
         :func:`vak.prep.segment_dataset.prep_segment_dataset`,
         and has a ``'split'`` column added.
     train_set_durs : list
         Durations in seconds of subsets taken from training data
         to create a learning curve, e.g., `[5., 10., 15., 20.]`.
     num_replicates : int
         number of times to replicate training for each training set duration
         to better estimate metrics for a training set of that size.
         Each replicate uses a different randomly drawn subset of the training
         data (but of the same duration).
     dataset_path : str, pathlib.Path
         Directory where splits will be saved.

     Returns
     -------
     dataset_df_out : pandas.DataFrame
         A pandas.DataFrame that has the original splits
         from ``dataset_df``, as well as the additional subsets
         of the training data added, along with additional
         columns, ``'subset', 'train_dur', 'replicate_num'``,
         that are used by :mod:`vak`.
         Other functions like :func:`vak.learncurve.learncurve`
         specify a specific subset of the training data
         by getting the subset name with the function
         :func:`vak.common.learncurve.get_train_dur_replicate_split_name`,
         and then filtering ``dataset_df_out`` with that name
         using the 'subset' column.
    """
    dataset_path = pathlib.Path(dataset_path)

    # get just train split, to pass to split.dataframe
    # so we don't end up with other splits in the training set
    train_split_df = dataset_df[dataset_df["split"] == "train"].copy()
    labelset = set([k for k in labelmap.keys() if k != "unlabeled"])

    # will concat after loop, then use ``csv_path`` to replace
    # original dataset df with this one
    subsets_df = []
    for train_dur in train_set_durs:
        logger.info(
            f"Subsetting training set for training set of duration: {train_dur}",
        )
        for replicate_num in range(1, num_replicates + 1):
            train_dur_replicate_subset_name = (
                common.learncurve.get_train_dur_replicate_subset_name(
                    train_dur, replicate_num
                )
            )

            train_dur_replicate_df = split.segment_dataframe(
                # copy to avoid mutating original train_split_df
                train_split_df.copy(),
                dataset_path,
                train_dur=train_dur,
                labelset=labelset,
            )
            # remove rows where split set to 'None'
            train_dur_replicate_df = train_dur_replicate_df[
                train_dur_replicate_df.split == "train"
            ]
            # next line, make split name in csv match the split name used for directory in dataset dir
            train_dur_replicate_df["subset"] = train_dur_replicate_subset_name
            train_dur_replicate_df["train_dur"] = train_dur
            train_dur_replicate_df["replicate_num"] = replicate_num
            subsets_df.append(train_dur_replicate_df)

    subsets_df = pd.concat(subsets_df)

    # keep the same validation, test, and total train sets by concatenating them with the train subsets
    dataset_df["subset"] = None  # add column but have it be empty
    dataset_df = pd.concat((subsets_df, dataset_df))
    # We reset the entire index across all splits, instead of repeating indices,
    # and we set drop=False because we don't want to add a new column 'index' or 'level_0'.
    # Need to do this again after calling `make_npy_files_for_each_split` since we just
    # did `pd.concat` with the original dataframe
    dataset_df = dataset_df.reset_index(drop=True)
    return dataset_df
