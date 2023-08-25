"""Functionality to prepare splits of frame classification datasets
to generate a learning curve."""
from __future__ import annotations

import logging
import pathlib
from typing import Sequence

import pandas as pd

from ... import common
from .. import split
from .dataset_arrays import make_npy_files_for_each_split


logger = logging.getLogger(__name__)


def make_learncurve_splits_from_dataset_df(
    dataset_df: pd.DataFrame,
    input_type: str,
    train_set_durs: Sequence[float],
    num_replicates: int,
    dataset_path: pathlib.Path,
    labelmap: dict,
    audio_format: str | None = None,
    spect_key: str = "s",
    timebins_key: str = "t",
) -> pd.DataFrame:
    """Make splits for a learning curve
    from a dataframe representing the entire dataset,
    one split for each combination of (training set duration,
    replicate number).
    Each split is a randomly drawn subset of data
    from the total training split.

    Uses :func:`vak.prep.split.frame_classification_dataframe` to make
    splits/subsets of the training data
    from ``dataset_df``, and then uses
    :func:`vak.prep.frame_classification.dataset_arrays.make_npy_files_for_each_split`
    to make the array files for each split.

    A new directory will be made for each combination of
    (training set duration, replicate number) as shown below,
    for ``train_durs=[4.0, 6.0], num_replicates=2``.

    .. code-block:: console
        032312-vak-frame-classification-dataset-generated-230820_144833
        ├── 032312_prep_230820_144833.csv
        ├── labelmap.json
        ├── metadata.json
        ├── prep_230820_144833.log
        ├── spectrograms_generated_230820_144833
        ├── test
        ├── train
        ├── train-dur-4.0-replicate-1
        ├── train-dur-4.0-replicate-2
        ├── train-dur-6.0-replicate-1
        ├── train-dur-6.0-replicate-2
        ├── TweetyNet_learncurve_audio_cbin_annot_notmat.toml
        └── val


    Parameters
    ----------
    dataset_df : pandas.DataFrame
        Representing an entire dataset of vocalizations.
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
        A :class:`dict` that maps a set of human-readable
        string labels to the integer classes predicted by a neural
        network model. As returned by :func:`vak.labels.to_map`.
    audio_format : str
        A :class:`string` representing the format of audio files.
        One of :constant:`vak.common.constants.VALID_AUDIO_FORMATS`.
    spect_key : str
        Key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        Key for accessing vector of time bins in files. Default is 't'.

    Returns
    -------
    dataset_df_out : pandas.DataFrame
        A pandas.DataFrame that has the original splits
        from ``dataset_df`` as well as the additional subsets
        of the training data added, along with additional
        'train_dur' and 'replicate_num' columns
        that can be used during analysis.
        Other functions like :func:`vak.learncurve.learncurve`
        specify a specific subset of the training data
        by getting the split name with the function
        :func:`vak.common.learncurve.get_train_dur_replicate_split_name`,
        and then filtering ``dataset_df_out`` with that name
        using the 'split' column.
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
            train_dur_replicate_split_name = (
                common.learncurve.get_train_dur_replicate_split_name(
                    train_dur, replicate_num
                )
            )

            train_dur_replicate_df = split.frame_classification_dataframe(
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
            train_dur_replicate_df["split"] = train_dur_replicate_split_name
            train_dur_replicate_df["train_dur"] = train_dur
            train_dur_replicate_df["replicate_num"] = replicate_num
            all_train_durs_and_replicates_df.append(train_dur_replicate_df)

    all_train_durs_and_replicates_df = pd.concat(
        all_train_durs_and_replicates_df
    )
    all_train_durs_and_replicates_df = make_npy_files_for_each_split(
        all_train_durs_and_replicates_df,
        dataset_path,
        input_type,
        "learncurve",  # purpose
        labelmap,
        audio_format,
        spect_key,
        timebins_key,
    )

    # keep the same validation, test, and total train sets by concatenating them with the train subsets
    dataset_df = pd.concat(
        (
            all_train_durs_and_replicates_df,
            dataset_df,
        )
    )
    # We reset the entire index across all splits, instead of repeating indices,
    # and we set drop=False because we don't want to add a new column 'index' or 'level_0'.
    # Need to do this again after calling `make_npy_files_for_each_split` since we just
    # did `pd.concat` with the original dataframe
    dataset_df = dataset_df.reset_index(drop=True)
    return dataset_df
