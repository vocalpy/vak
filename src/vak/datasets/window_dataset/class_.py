from __future__ import annotations

import pathlib
from typing import Callable

import crowsetta
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torchvision.datasets.vision import VisionDataset

from ... import transforms
from ...common import (
    annotation,
    files,
    validators
)
from ..metadata import Metadata
from .helper import vectors_from_df


def get_window_inds(n_frames: int, window_size: int, stride: int):
    return np.arange(stop=n_frames - (window_size - 1), step=stride)


class FrameClassificationWindowDataset:
    """Dataset used for training neural network models
    on the frame classification task.
    where the source data consists of audio signals
    or spectrograms of varying lengths.

    Unlike
    :class:`vak.datasets.frame_classification.FrameClassificationEvalDataset`,
    this class does not return entire samples
    from the source dataset.
    Instead each paired samples :math:`(x_i, y_i)`
    returned by this dataset class consists of
    a window :math:`x_i` of fixed length
    :math:`w` from the underlying data ``X`` of total length :math:`T`.
    Each :math:`y_i` is a vector of the same size :math:`w`, containing
    an integer class label for each *frame* in the window :math:`x_i`.
    The entire dataset consists of some number of windows
    :math:`I` determined by a ``stride`` parameter :math:`s`,
    :math:`I = (T - w) / s`.

    The underlying data consists of single arrays
    for both the input to the network ``X``
    and the targets for the network output ``Y``.
    These single arrays ``X`` and ``Y`` are
    created by concatenating samples from the source
    data, e.g., audio files or spectrogram arrays.
    (This is true for
    :class:`vak.datasets.frame_classification.FrameClassificationEvalDataset`
    as well.)
    The dimensions of :math:`X`  will be (channels, ..., frames),
    i.e., audio will have dimensions (channels, samples)
    and spectrograms will have dimensions
    (channels, frequency bins, time bins).
    The signal :math:`X` may be either audio or spectrogram,
    meaning that a frame will be either a single sample
    in an audio signal or a single time bin in a spectrogram.
    The last dimension of ``X`` will always be the
    number of total frames in the dataset,
    either audio samples or spectrogram time bins,
    and ``Y`` will be the same size, containing
    an integer class label for each frame.
    """

    def __init__(
            self,
            X: npt.NDArray,
            Y: npt.NDArray,
            window_size: int,
            stride: int = 1,
            window_inds: npt.NDArray | None = None,
            transform: Callable | None = None,
            target_transform: Callable | None = None
    ):
        self.window_size = window_size
        self.stride = stride
        self.X = X

        if window_inds is None:
            window_inds = get_window_inds(X.shape[-1], window_size, stride)
        self.window_inds = window_inds
        self.Y = Y
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        arr_idx = self.window_inds[idx]
        x = self.X[..., arr_idx:arr_idx + self.window_size]
        y = self.Y[arr_idx:arr_idx + self.window_size]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        """number of batches"""
        return len(self.window_inds)

    @classmethod
    def from_dataset_path(
            cls,
            dataset_path: str | pathlib.Path,
            window_size: int,
            stride: int = 1,
            split: str = "train",
            transform: Callable | None = None,
            target_transform: Callable | None = None
    ):
        dataset_path = pathlib.Path(dataset_path)
        split_path = dataset_path / split
        X_path = split_path / 'X_T.npy'
        X = np.load(X_path)
        Y_path = split_path / 'Y_T.npy'
        Y = np.load(Y_path)
        window_inds_path = split_path / 'window_inds.npy'
        if window_inds_path.exists():
            window_inds = np.load(window_inds_path)
        else:
            window_inds = None
        return cls(
            X,
            Y,
            window_size,
            stride,
            window_inds,
            transform,
            target_transform
        )
