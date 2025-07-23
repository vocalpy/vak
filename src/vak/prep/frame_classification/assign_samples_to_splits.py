from __future__ import annotations

import logging
import pathlib
import re
from collections import defaultdict

import pandas as pd

from .. import dataset_df_helper, split

logger = logging.getLogger(__name__)


def extract_prefix(filename: str) -> str:
    """Extracts session prefix from a filename.
    Adjust regex if needed to suit your naming convention."""
    return re.sub(r'_S\d+\.wav$', '', filename)


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
    assign each sample (row) to a split, respecting session-level grouping.

    If no durations are specified for splits,
    or the purpose is either `'eval'` or `'predict'`,
    then all rows in the dataframe will be assigned to ``purpose``.

    Parameters
    ----------
    purpose : str
        One of {'train', 'eval', 'predict', 'learncurve'}.
    dataset_df : pandas.DataFrame
        That represents a dataset.
    dataset_path : pathlib.Path
        Path to csv saved from ``dataset_df``.
    train_dur : float
        Duration of training set (seconds). Default is None.
    val_dur : float
        Duration of validation set (seconds). Default is None.
    test_dur : float
        Duration of test set (seconds). Default is None.
    labelset : set | None
        Labels to use. If given, will drop samples with other labels.

    Returns
    -------
    dataset_df : pandas.DataFrame
        With a new `'split'` column assigning each sample to a dataset split.
    """
    if (purpose == "train" or purpose == "learncurve") and (
        (train_dur is not None and train_dur > 0)
        and (val_dur is None or val_dur == 0)
        and (test_dur is None or test_dur == 0)
    ):
        raise ValueError(
            "A duration specified for just training set, but prep function does not currently support creating a "
            "single split of a specified duration. Either remove the train_dur option from the prep section and "
            "rerun, in which case all data will be included in the training set, or specify values greater than "
            "zero for test_dur (and val_dur, if a validation set will be used)"
        )

    if all([dur is None for dur in (train_dur, val_dur, test_dur)]) or purpose in ("eval", "predict"):
        logger.info("Will not split dataset.")
        split_name = "test" if purpose == "eval" else "predict" if purpose == "predict" else purpose
        return dataset_df_helper.add_split_col(dataset_df, split=split_name)

    logger.info("Will split dataset.")

    MAX_FILE_DURATION_FOR_ASSESS = 100  # seconds
    short_df = dataset_df[dataset_df["duration"] <= MAX_FILE_DURATION_FOR_ASSESS].copy()
    long_df = dataset_df[dataset_df["duration"] > MAX_FILE_DURATION_FOR_ASSESS].copy()

    long_dur = long_df["duration"].sum()
    remaining_train_dur = max(train_dur - long_dur, 0)

    short_total = short_df["duration"].sum()
    required_short_dur = remaining_train_dur + (val_dur or 0) + (test_dur or 0)

    if short_total < required_short_dur:
        raise ValueError(
            f"Insufficient short files to meet split durations: need {required_short_dur}s, "
            f"but only {short_total}s available."
        )

    logger.info("Using session-aware splitting strategy.")
    short_df = short_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle for fairness
    short_df["prefix"] = short_df["audio_path"].apply(lambda p: extract_prefix(pathlib.Path(p).name))

    grouped = defaultdict(list)
    for _, row in short_df.iterrows():
        grouped[row["prefix"]].append(row)

    train_rows, val_rows, test_rows = [], [], []
    durs = {"train": remaining_train_dur, "val": val_dur or 0, "test": test_dur or 0}

    for prefix, rows in grouped.items():
        session_rows = pd.DataFrame(rows)
        session_rows["split"] = None
        session_dur = session_rows["duration"].sum()

        if len(session_rows) == 1:
            target = max(durs, key=lambda k: durs[k])  # assign where we need more
        elif len(session_rows) == 2:
            targets = sorted(durs, key=durs.get, reverse=True)[:2]
            session_rows.iloc[0, session_rows.columns.get_loc("split")] = targets[0]
            session_rows.iloc[1, session_rows.columns.get_loc("split")] = targets[1]
            train_rows.append(session_rows[session_rows["split"] == "train"])
            val_rows.append(session_rows[session_rows["split"] == "val"])
            test_rows.append(session_rows[session_rows["split"] == "test"])
            continue
        else:
            targets = ["train", "val", "test"]
            for i, target in enumerate(targets):
                durs[target] -= session_rows.iloc[i]["duration"]
                session_rows.iloc[i, session_rows.columns.get_loc("split")] = target
            for i in range(3, len(session_rows)):
                session_rows.iloc[i, session_rows.columns.get_loc("split")] = "train"
                durs["train"] -= session_rows.iloc[i]["duration"]
            train_rows.append(session_rows[session_rows["split"] == "train"])
            val_rows.append(session_rows[session_rows["split"] == "val"])
            test_rows.append(session_rows[session_rows["split"] == "test"])
            continue

        durs[target] -= session_dur
        session_rows["split"] = target
        if target == "train":
            train_rows.append(session_rows)
        elif target == "val":
            val_rows.append(session_rows)
        else:
            test_rows.append(session_rows)

    final_df = pd.concat(train_rows + val_rows + test_rows, ignore_index=True)
    final_df.drop(columns=["prefix"], inplace=True)

    long_df = dataset_df_helper.add_split_col(long_df, split="train")
    dataset_df = pd.concat([final_df, long_df], axis=0, ignore_index=True)

    return dataset_df
