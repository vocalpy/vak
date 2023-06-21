"""Functionality to prepare splits of frame classification datasets
to generate a learning curve."""
from __future__ import annotations

import logging
import pathlib
from typing import Sequence

import crowsetta
import numpy as np
import pandas as pd

from .dataset_arrays import (
    sort_source_paths_and_annots_by_label_freq,
    make_from_source_paths_and_annots
)
from .. import split
from ... import common, datasets


logger = logging.getLogger(__name__)


def make_learncurve_splits_from_dataset_df(
    dataset_df: pd.DataFrame,
    csv_path: str | pathlib.Path,
    input_type: str,
    train_set_durs: Sequence[float],
    num_replicates: int,
    dataset_path: pathlib.Path,
    labelmap: dict,
    audio_format: str | None = None,
    spect_key: str = "s",
    timebins_key: str = "t",
):
    """Make splits for a learning curve from a dataframe representing the entire dataset.

    Uses :func:`vak.split.dataframe` to make splits from ``dataset_df``.
    Then makes a new directory inside ``dataset_path`` for each training set split,
    one split for each combination of training set duration and replicate number.
    In each directory, it saves the three array files representing the frame
    classification dataset, produced by calling
    :func:`vak.prep.frame_classification.helper.make_frame_classification_arrays_from_source_paths_and_annots`.

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        Representing an entire dataset of vocalizations.
    csv_path : pathlib.Path
        Path to where dataset was saved as a csv file.
    input_type : str
        The type of input to the neural network model.
        One of {'audio', 'spect'}.
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
    labelmap : dict
        that maps labelset to consecutive integers
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    """
    dataset_path = pathlib.Path(dataset_path)

    # get just train split, to pass to split.dataframe
    # so we don't end up with other splits in the training set
    train_split_df = dataset_df[dataset_df["split"] == "train"].copy()
    labelset = set([k for k in labelmap.keys() if k != "unlabeled"])

    # will concat after loop, then use ``csv_path`` to replace
    # original dataset df with this one
    all_train_durs_and_replicates_df = []
    for train_dur in train_set_durs:
        logger.info(
            f"Subsetting training set for training set of duration: {train_dur}",
        )
        for replicate_num in range(1, num_replicates + 1):
            train_dur_replicate_split_name = common.learncurve.get_train_dur_replicate_split_name(
                train_dur, replicate_num
            )

            train_dur_replicate_df = split.dataframe(
                # copy to avoid mutating original train_split_df
                train_split_df.copy(), dataset_path, train_dur=train_dur, labelset=labelset
            )
            # remove rows where split set to 'None'
            train_dur_replicate_df = train_dur_replicate_df[train_dur_replicate_df.split == "train"]
            # next line, make split name in csv math split name used for directory in dataset dir
            train_dur_replicate_df['split'] = train_dur_replicate_split_name
            train_dur_replicate_df['train_dur'] = train_dur
            train_dur_replicate_df['replicate_num'] = replicate_num
            all_train_durs_and_replicates_df.append(
                train_dur_replicate_df
            )

            metadata = datasets.metadata.Metadata.from_dataset_path(dataset_path)
            frame_dur = metadata.frame_dur

            if input_type == 'audio':
                source_paths = train_dur_replicate_df['audio_path'].values
            elif input_type == 'spect':
                source_paths = train_dur_replicate_df['spect_path'].values
            else:
                raise ValueError(
                    f"Invalid ``input_type``: {input_type}"
                )
            annots = common.annotation.from_df(train_dur_replicate_df)

            # sort to minimize chance that cropping removes classes
            source_paths, annots = sort_source_paths_and_annots_by_label_freq(
                source_paths,
                annots
            )

            (inputs,
             source_id_vec,
             frame_labels) = make_from_source_paths_and_annots(
                source_paths,
                input_type,
                annots,
                labelmap,
                train_dur,
                frame_dur,
                audio_format,
                spect_key,
                timebins_key,
            )

            train_dur_replicate_root = dataset_path / train_dur_replicate_split_name
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
    all_train_durs_and_replicates_df = pd.concat(all_train_durs_and_replicates_df)
    all_train_durs_and_replicates_df = pd.concat(
        (
            all_train_durs_and_replicates_df,
            dataset_df[dataset_df.split == "val"],
            dataset_df[dataset_df.split == "test"],
        )
    )

    all_train_durs_and_replicates_df.to_csv(csv_path, index=False)
