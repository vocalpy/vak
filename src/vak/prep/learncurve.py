"""Functions that return a dict mapping training set durations to csv paths,
used by ``vak.core.learncurve``"""
from __future__ import annotations

import logging
import pathlib
from typing import Sequence

import crowsetta
import numpy as np
import pandas as pd

from . import split
from .. import common, datasets
from .frame_classification.helper import (
    sort_source_paths_and_annots_by_label_freq,
    make_frame_classification_arrays_from_spect_and_annot_paths
)


logger = logging.getLogger(__name__)


def make_learncurve_splits_from_dataset_df(
    dataset_df: pd.DataFrame,
    csv_path: str | pathlib.Path,
    train_set_durs: Sequence[float],
    num_replicates: int,
    dataset_path: pathlib.Path,
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

            metadata = datasets.metadata.Metadata.from_dataset_path(dataset_path)
            timebin_dur = metadata.timebin_dur

            source_paths = train_split_df['spect_path'].values
            annots = common.annotation.from_df(train_split_df)

            source_paths, annots = sort_source_paths_and_annots_by_label_freq(
                source_paths,
                annots
            )

            (inputs,
             source_id_vec,
             frame_labels) = make_frame_classification_arrays_from_spect_and_annot_paths(
                source_paths,
                labelmap,
                annots,
                train_dur,
                timebin_dur,
            )

            train_dur_replicate_root = learncurve_splits_root / f"train-dur-{train_dur}-replicate-{replicate_num}"
            train_dur_replicate_root.mkdir()

            logger.info(
                "Saving ``inputs`` vector for frame classification dataset with size "
                f"{round(inputs.nbytes * 1e-6, 2)} MB."
            )
            np.save(train_dur_replicate_root / datasets.frame_classification.constants.INPUT_ARRAY_FILENAME, inputs)
            logger.info(
                "Saving ``source_id`` vector for frame classification dataset with size "
                f"{round(source_id_vec.nbytes * 1e-6, 2)} MB."
            )
            np.save(train_dur_replicate_root / datasets.frame_classification.constants.SOURCE_IDS_ARRAY_FILENAME,
                    source_id_vec)
            logger.info(
                "Saving ``frame_labels`` vector (targets) for frame classification dataset "
                f"with size {round(frame_labels.nbytes * 1e-6, 2)} MB."
            )
            np.save(train_dur_replicate_root / datasets.frame_classification.constants.FRAME_LABELS_ARRAY_FILENAME,
                    frame_labels)
            logger.info(
                "Saving csv file of annotations for frame classification dataset"
            )
            generic_seq = crowsetta.formats.seq.GenericSeq(annots=annots)
            generic_seq.to_file(
                train_dur_replicate_root / datasets.frame_classification.constants.ANNOTATION_CSV_FILENAME
            )

            # keep the same validation and test set by concatenating them with the train subset
            split_df = pd.concat(
                (
                    train_split_df,
                    dataset_df[dataset_df.split == "val"],
                    dataset_df[dataset_df.split == "test"],
                )
            )

            split_csv_filename = f"{csv_path.stem}-train-dur-{train_dur}s-replicate-{replicate_num}.csv"
            # note that learncurve split csvs are in dataset_path, not dataset_learncurve_dir
            # this is to avoid changing the semantics of dataset_csv_path to other functions that expect it,
            # e.g., ``StandardizeSpect.fit_csv_path``; any dataset csv path always has to be in the root
            split_csv_path = dataset_path / split_csv_filename
            split_df.to_csv(split_csv_path, index=False)
            # save just name, will load relative to dataset_path
            record['split_csv_filename'] = split_csv_path.name

            splits_records.append(record)

    splits_df = pd.DataFrame.from_records(splits_records)
    splits_path = learncurve_splits_root / 'learncurve-splits-metadata.csv'
    splits_df.to_csv(splits_path, index=False)
