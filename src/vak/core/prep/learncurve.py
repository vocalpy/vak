"""Functions that return a dict mapping training set durations to csv paths,
used by ``vak.core.learncurve``"""
from __future__ import annotations

import logging
import pathlib
from typing import Sequence

import numpy as np
import pandas as pd

from .prep_helper import validate_and_get_timebin_dur
from ... import split
from ...datasets import window_dataset


logger = logging.getLogger(__name__)


def make_learncurve_splits_from_dataset_df(
    dataset_df: pd.DataFrame,
    csv_path: str | pathlib.Path,
    train_set_durs: Sequence[float],
    num_replicates: int,
    dataset_path: pathlib.Path,
    window_size: int,
    labelmap: dict,
    spect_key: str = "s",
    timebins_key: str = "t",
):
    """Make splits for a learning curve from a dataframe representing the entire dataset.

    Uses :func:`vak.split.dataframe` to make splits from ``dataset_df``.
    Makes a new directory named "learncurve" in the root of the directory ``dataset_path``,
    and then saves the following in that directory:
    - A csv file for each split, representing one replicate of one training set duration
    - Three npy files for each split, the vectors representing the window dataset
    - A json file that maps training set durations to replicate numbers,
      and replicate numbers to the path to the csv, relative to dataset_path

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        Representing an entire dataset of vocalizations.
    csv_path : pathlib.Path
        path to where dataset was saved as a csv.
    train_set_durs : list
        of int, durations in seconds of subsets taken from training data
        to create a learning curve, e.g. [5, 10, 15, 20].
    num_replicates : int
        number of times to replicate training for each training set duration
        to better estimate metrics for a training set of that size.
        Each replicate uses a different randomly drawn subset of the training
        data (but of the same duration).
    dataset_path : str, pathlib.Path
        Directory where splits will be saved.
    window_size : int
        Size of windows taken from spectrograms, in number of time bins,
        shown to neural networks
    labelmap : dict
        that maps labelset to consecutive integers
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    """
    dataset_path = pathlib.Path(dataset_path)
    learncurve_splits_root = dataset_path / 'learncurve'
    learncurve_splits_root.mkdir()

    splits_records = []  # will use to create dataframe, then save as csv
    for train_dur in train_set_durs:
        logger.info(
            f"Subsetting training set for training set of duration: {train_dur}",
        )
        for replicate_num in range(1, num_replicates + 1):
            record = {
                'train_dur': train_dur,
                'replicate_num': replicate_num,
            }  # will add key-val pairs to this, then append to splits_records at end of inner loop

            # get just train split, to pass to split.dataframe
            # so we don't end up with other splits in the training set
            train_split_df = dataset_df[dataset_df["split"] == "train"]
            labelset = set([k for k in labelmap.keys() if k != "unlabeled"])
            train_split_df = split.dataframe(
                train_split_df, dataset_path, train_dur=train_dur, labelset=labelset
            )
            train_split_df = train_split_df[train_split_df.split == "train"]  # remove rows where split set to 'None'

            timebin_dur = validate_and_get_timebin_dur(dataset_df)
            # use *just* train subset to get spect vectors for WindowDataset
            (
                source_ids,
                source_inds,
                window_inds,
            ) = window_dataset.helper.vectors_from_df(
                train_split_df,
                "train",
                window_size,
                spect_key,
                timebins_key,
                crop_dur=train_dur,
                timebin_dur=timebin_dur,
                labelmap=labelmap,
            )

            # TODO: this is specific to WindowDataset -- flag? separate learncurve split functions for other datasets?
            for vec_name, vec in zip(
                ["source_ids", "source_inds", "window_inds"],
                [source_ids, source_inds, window_inds],
            ):
                vector_path = learncurve_splits_root / f"{vec_name}-train-dur-{train_dur}-replicate-{replicate_num}.npy"
                np.save(str(vector_path), vec)  # str so type-checker doesn't complain
                # save just name, will load relative to dataset_path
                record[f'{vec_name}_npy_filename'] = vector_path.name

            # keep the same validation and test set by concatenating them with the train subset
            split_df = pd.concat(
                (
                    train_split_df,
                    dataset_df[dataset_df.split == "val"],
                    dataset_df[dataset_df.split == "test"],
                )
            )

            split_csv_filename = f"{csv_path.stem}-train-dur-{train_dur}s-replicate-{replicate_num}.csv"
            split_csv_path = learncurve_splits_root / split_csv_filename
            split_df.to_csv(split_csv_path, index=False)
            # save just name, will load relative to dataset_path
            record['split_csv_filename'] = split_csv_path.name

            splits_records.append(record)

    splits_df = pd.DataFrame.from_records(splits_records)
    splits_path = learncurve_splits_root / 'learncurve-splits-metadata.csv'
    splits_df.to_csv(splits_path, index=False)
