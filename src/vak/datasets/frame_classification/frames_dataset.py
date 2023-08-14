from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import constants
from .metadata import Metadata


class FramesDataset:
    """A dataset class used for
    neural network models
    with the frame classification task,
    where the source data consists of audio signals
    or spectrograms of varying lengths.

    Attributes
    ----------
    dataset_path
    dataset_df
    frame_dur : float
        Duration of a single frame, in seconds.
    duration : float
        Total duration of the dataset.
    """

    def __init__(
        self,
        dataset_path: str | pathlib.Path,
        dataset_df: pd.DataFrame,
        split: str,
        sample_ids: npt.NDArray,
        inds_in_sample: npt.NDArray,
        frame_dur: float,
        input_type: str,
        item_transform: Callable | None = None,
    ):
        self.dataset_path = pathlib.Path(dataset_path)

        self.split = split
        dataset_df = dataset_df[dataset_df.split == split].copy()
        self.dataset_df = dataset_df
        self.frames_paths = self.dataset_df[
            constants.FRAMES_NPY_PATH_COL_NAME
        ].values
        if split != "predict":
            self.frame_labels_paths = self.dataset_df[
                constants.FRAME_LABELS_NPY_PATH_COL_NAME
            ].values
        else:
            self.frame_labels_paths = None

        if input_type == "audio":
            self.source_paths = self.dataset_df["audio_path"].values
        elif input_type == "spect":
            self.source_paths = self.dataset_df["spect_path"].values
        else:
            raise ValueError(
                f"Invalid `input_type`: {input_type}. Must be one of {{'audio', 'spect'}}."
            )

        self.sample_ids = sample_ids
        self.inds_in_sample = inds_in_sample
        self.frame_dur = float(frame_dur)
        self.item_transform = item_transform

    @property
    def duration(self):
        return self.sample_ids.shape[-1] * self.frame_dur

    @property
    def shape(self):
        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        return tmp_item["frames"].shape

    def __getitem__(self, idx):
        source_path = self.source_paths[idx]
        frames = np.load(self.dataset_path / self.frames_paths[idx])
        item = {"frames": frames, "source_path": source_path}
        if self.frame_labels_paths is not None:
            frame_labels = np.load(
                self.dataset_path / self.frame_labels_paths[idx]
            )
            item["frame_labels"] = frame_labels

        if self.item_transform:
            item = self.item_transform(**item)

        return item

    def __len__(self):
        """number of batches"""
        return len(np.unique(self.sample_ids))

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str | pathlib.Path,
        split: str = "val",
        item_transform: Callable | None = None,
    ):
        """

        Parameters
        ----------
        dataset_path
        split
        item_transform

        Returns
        -------

        """
        dataset_path = pathlib.Path(dataset_path)
        metadata = Metadata.from_dataset_path(dataset_path)
        frame_dur = metadata.frame_dur
        input_type = metadata.input_type

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)

        split_path = dataset_path / split
        sample_ids_path = split_path / constants.SAMPLE_IDS_ARRAY_FILENAME
        sample_ids = np.load(sample_ids_path)
        inds_in_sample_path = (
            split_path / constants.INDS_IN_SAMPLE_ARRAY_FILENAME
        )
        inds_in_sample = np.load(inds_in_sample_path)

        return cls(
            dataset_path,
            dataset_df,
            split,
            sample_ids,
            inds_in_sample,
            frame_dur,
            input_type,
            item_transform,
        )
