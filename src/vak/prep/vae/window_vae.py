from __future__ import annotations

import json
import logging
import pathlib

import pandas as pd

from ...common import labels
from .. import sequence_dataset
from ..spectrogram_dataset import prep_spectrogram_dataset
from ..frame_classification.assign_samples_to_splits import assign_samples_to_splits
from ..frame_classification.learncurve import make_subsets_from_dataset_df
from ..frame_classification.make_splits import make_splits


logger = logging.getLogger(__name__)


def prep_window_vae_dataset(
    data_dir: str | pathlib.Path,
    dataset_path: str | pathlib.Path,
    dataset_csv_path: str | pathlib.Path,
    purpose: str,
    audio_format: str | None = None,
    spect_format: str | None = None,
    spect_params: dict | None = None,
    spect_output_dir: str | pathlib.Path | None = None,
    annot_format: str | None = None,
    annot_file: str | pathlib.Path | None = None,
    labelset: set | None = None,
    audio_dask_bag_kwargs: dict | None = None,
    train_dur: int | None = None,
    val_dur: int | None = None,
    test_dur: int | None = None,
    train_set_durs: list[float] | None = None,
    num_replicates: int | None = None,
    spect_key: str = "s",
    timebins_key: str = "t",
    freqbins_key: str = "f",
) -> pd.DataFrame:
    """

    Parameters
    ----------
    data_dir
    dataset_path
    dataset_csv_path
    purpose
    audio_format
    spect_format
    spect_params
    spect_output_dir
    annot_format
    annot_file
    labelset
    audio_dask_bag_kwargs
    train_dur
    val_dur
    test_dur
    train_set_durs
    num_replicates
    spect_key
    timebins_key

    Returns
    -------

    """
    source_files_df = prep_spectrogram_dataset(
        data_dir,
        annot_format,
        labelset,
        annot_file,
        audio_format,
        spect_format,
        spect_params,
        spect_output_dir,
        audio_dask_bag_kwargs,
    )

    # save before (possibly) splitting, just in case duration args are not valid
    # (we can't know until we make dataset)
    source_files_df.to_csv(dataset_csv_path)

    # ---- assign samples to splits; adds a 'split' column to dataset_df, calling `vak.prep.split` if needed -----------
    # once we assign a split, we consider this the ``dataset_df``
    dataset_df: pd.DataFrame = assign_samples_to_splits(
        purpose,
        source_files_df,
        dataset_path,
        train_dur,
        val_dur,
        test_dur,
        labelset,
    )

    # ---- create and save labelmap ------------------------------------------------------------------------------------
    # we do this before creating array files since we need to load the labelmap to make frame label vectors
    if purpose != "predict":
        # TODO: add option to generate predict using existing dataset, so we can get labelmap from it
        map_unlabeled_segments = sequence_dataset.has_unlabeled_segments(
            dataset_df
        )
        labelmap = labels.to_map(
            labelset, map_unlabeled=map_unlabeled_segments
        )
        logger.info(
            f"Number of classes in labelmap: {len(labelmap)}",
        )
        # save labelmap in case we need it later
        with (dataset_path / "labelmap.json").open("w") as fp:
            json.dump(labelmap, fp)
    else:
        labelmap = None

    # ---- actually move/copy/create files into directories representing splits ----------------------------------------
    # now we're *remaking* the dataset_df (actually adding additional rows with the splits)
    dataset_df: pd.DataFrame = make_splits(
        dataset_df,
        dataset_path,
        # input_type="spect", we only make spectrogram datasets for now
        "spect",
        purpose,
        labelmap,
        audio_format,
        spect_key,
        timebins_key,
        freqbins_key,
    )

    # ---- if purpose is learncurve, additionally prep training data subsets for the learning curve --------------------
    if purpose == "learncurve":
        dataset_df: pd.DataFrame = make_subsets_from_dataset_df(
            dataset_df,
            input_type,
            train_set_durs,
            num_replicates,
            dataset_path,
            labelmap,
        )

    # ---- save csv file that captures provenance of source data -------------------------------------------------------
    logger.info(f"Saving dataset csv file: {dataset_csv_path}")
    dataset_df.to_csv(
        dataset_csv_path, index=False
    )  # index is False to avoid having "Unnamed: 0" column when loading

    return dataset_df
