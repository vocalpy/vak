from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.typing as npt

from . import constants
from ..metadata import Metadata


def get_window_inds(n_frames: int, window_size: int, stride: int = 1):
    """Get indices of windows for a :class:`WindowDataset`,
    given the number of frames in the dataset,
    the window size, and the stride.

    This function is used by :class:`WindowDataset`
    to compute the indices of windows in the dataset.
    The length of the vector of indices it returns
    is the number of windows in the dataset,
    i.e., the number of samples.

    Parameters
    ----------
    n_frames : int
    window_size : int
    stride : int

    Returns
    -------
    window_inds : numpy.ndarray
        Vector of indices for windows.
        During training, batches of windows are made
        by grabbing indices randomly from this vector,
        then getting windows of the specified size
        from the arrays representing the input data
        and targets for the neural network.
    """
    return np.arange(stop=n_frames - (window_size - 1), step=stride)


class WindowDataset:
    """Dataset used for training neural network models
    on the frame classification task.
    where the source data consists of audio signals
    or spectrograms of varying lengths.

    Unlike
    :class:`vak.datasets.frame_classification.FramesDataset`,
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
    :class:`vak.datasets.frame_classification.FramesDataset`
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

    Attributes
    ----------
    X : numpy.ndarray
    Y : numpy.ndarray
    window_size : int
    frame_dur : float
        Duration of a single frame, in seconds.
    duration : float
        Total duration of the dataset.
    """

    def __init__(
            self,
            X: npt.NDArray,
            Y: npt.NDArray,
            window_size: int,
            frame_dur: float,
            stride: int = 1,
            window_inds: npt.NDArray | None = None,
            transform: Callable | None = None,
            target_transform: Callable | None = None
    ):
        self.X = X
        self.Y = Y
        self.window_size = window_size
        self.frame_dur = float(frame_dur)
        self.stride = stride

        if window_inds is None:
            window_inds = get_window_inds(X.shape[-1], window_size, stride)
        self.window_inds = window_inds

        self.transform = transform
        self.target_transform = target_transform

    @property
    def duration(self):
        return self.X.shape[-1] * self.frame_dur

    @property
    def shape(self):
        tmp_x_ind = 0
        one_x, _ = self.__getitem__(tmp_x_ind)
        # used by vak functions that need to determine size of window,
        # e.g. when initializing a neural network model
        return one_x.shape

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
        X_path = split_path / constants.INPUT_ARRAY_FILENAME
        X = np.load(X_path)
        Y_path = split_path / constants.FRAME_LABELS_ARRAY_FILENAME
        Y = np.load(Y_path)
        window_inds_path = split_path / constants.WINDOW_INDS_ARRAY_FILENAME
        if window_inds_path.exists():
            window_inds = np.load(window_inds_path)
        else:
            window_inds = None
        metadata = Metadata.from_dataset_path(dataset_path)
        frame_dur = metadata.timebin_dur

        return cls(
            X,
            Y,
            window_size,
            frame_dur,
            stride,
            window_inds,
            transform,
            target_transform
        )
