from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import constants
from ..metadata import Metadata


class FramesDataset:
    """A dataset class used for
    neural network models
    with the frame classification task,
    where the source data consists of audio signals
    or spectrograms of varying lengths.

    The underlying data consists of single arrays
    for both the input to the network ``X``
    and the targets for the network output ``Y``.
    These single arrays ``X`` and ``Y`` are
    created by concatenating samples from the source
    data, e.g., audio files or spectrogram arrays.
    The last dimension of ``X`` will always be the
    number of total frames in the dataset,
    either audio samples or spectrogram time bins,
    and ``Y`` will be the same size, containing
    an integer class label for each frame.

    To recover the original samples from the
    concatenated array, a ``sample_ids`` vector
    is used with integer elements
    :math:`(0, 0, 0, ..., 1, 1, 1, ..., i, i, i)```
    for a dataset with :math:`M` samples
    :math:`(1, 2, ..., i)`.
    The vector ``sample_ids`` will be the same length
    as ``X`` and ``Y``. Each element in ``sample_ids``
    indicates for each frame in the total dataset
    which original sample that frame belongs to.

    Attributes
    ----------
    dataset_df
    Y : numpy.ndarray
    frame_dur : float
        Duration of a single frame, in seconds.
    duration : float
        Total duration of the dataset.

    Examples
    --------

    This snippet demonstrates how samples are recovered by array indexing.
    In the snippet we get the first sample using its id number, ``0``
    (because Python uses zero indexing).
    This is basically what :meth:`FramesDataset.__getitem__` does
    when called with an ``idx``.

    >>> dataset = FramesDataset(X, Y, sample_ids)
    >>> x1, y1 = dataset.X[dataset.sample_ids == 0], dataset.Y[dataset.sample_ids == 0]
    """
    def __init__(
        self,
        dataset_path: str | pathlib.Path,
        dataset_df: pd.DataFrame,
        sample_ids: npt.NDArray,
        inds_in_sample: npt.NDArray,
        frame_dur: float,
        item_transform: Callable | None = None,
    ):
        self.dataset_path = pathlib.Path(dataset_path)

        self.dataset_df = dataset_df
        self.frames_paths = dataset_df[constants.FRAMES_NPY_PATH_COL_NAME].values
        self.frame_labels_paths = dataset_df[constants.FRAME_LABELS_NPY_PATH_COL_NAME].values

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
        frames = np.load(self.dataset_path / self.frames_paths[idx])
        frame_labels = np.load(self.dataset_path / self.frame_labels_paths[idx])

        if self.item_transform:
            item = self.item_transform(frames, frame_labels)
        else:
            item = {
                "frames": frames,
                "frame_labels": frame_labels,
            }

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
        dataset_path = pathlib.Path(dataset_path)
        metadata = Metadata.from_dataset_path(dataset_path)
        frame_dur = metadata.frame_dur

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)

        split_path = dataset_path / split
        sample_ids_path = split_path / constants.SAMPLE_IDS_ARRAY_FILENAME
        sample_ids = np.load(sample_ids_path)
        inds_in_sample_path = split_path / constants.INDS_IN_SAMPLE_ARRAY_FILENAME
        inds_in_sample = np.load(inds_in_sample_path)

        return cls(
            dataset_path,
            dataset_df,
            sample_ids,
            inds_in_sample,
            frame_dur,
            item_transform,
        )
