from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import constants
from .metadata import Metadata


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
        dataset_path: str | pathlib.Path,
        dataset_df: pd.DataFrame,
        split: str,
        sample_ids: npt.NDArray,
        inds_in_sample: npt.NDArray,
        window_size: int,
        frame_dur: float,
        stride: int = 1,
        window_inds: npt.NDArray | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.dataset_path = pathlib.Path(dataset_path)

        self.split = split
        dataset_df = dataset_df[dataset_df.split == split].copy()
        self.dataset_df = dataset_df

        self.frames_paths = self.dataset_df[
            constants.FRAMES_NPY_PATH_COL_NAME
        ].values
        self.frame_labels_paths = self.dataset_df[
            constants.FRAME_LABELS_NPY_PATH_COL_NAME
        ].values

        self.sample_ids = sample_ids
        self.inds_in_sample = inds_in_sample

        self.window_size = window_size
        self.frame_dur = float(frame_dur)
        self.stride = stride

        if window_inds is None:
            window_inds = get_window_inds(
                sample_ids.shape[-1], window_size, stride
            )
        self.window_inds = window_inds

        self.transform = transform
        self.target_transform = target_transform

    @property
    def duration(self):
        return self.sample_ids.shape[-1] * self.frame_dur

    @property
    def shape(self):
        tmp_x_ind = 0
        one_x, _ = self.__getitem__(tmp_x_ind)
        # used by vak functions that need to determine size of window,
        # e.g. when initializing a neural network model
        return one_x.shape

    def __getitem__(self, idx):
        window_idx = self.window_inds[idx]
        sample_ids = self.sample_ids[
            window_idx: window_idx + self.window_size
        ]
        uniq_sample_ids = np.unique(sample_ids)
        if len(uniq_sample_ids) == 1:
            sample_id = uniq_sample_ids[0]
            frames = np.load(self.dataset_path / self.frames_paths[sample_id])
            frame_labels = np.load(
                self.dataset_path / self.frame_labels_paths[sample_id]
            )
        elif len(uniq_sample_ids) > 1:
            frames = []
            frame_labels = []
            for sample_id in sorted(uniq_sample_ids):
                frames.append(
                    np.load(self.dataset_path / self.frames_paths[sample_id])
                )
                frame_labels.append(
                    np.load(
                        self.dataset_path / self.frame_labels_paths[sample_id]
                    )
                )

            if all([frames_.ndim == 1 for frames_ in frames]):
                # --> all 1-d audio vectors; if we specify `axis=1` here we'd get error
                frames = np.concatenate(frames)
            else:
                frames = np.concatenate(frames, axis=1)
            frame_labels = np.concatenate(frame_labels)
        else:
            raise ValueError(
                f"Unexpected number of ``uniq_sample_ids``: {uniq_sample_ids}"
            )

        inds_in_sample = self.inds_in_sample[window_idx]
        frames = frames[
            ..., inds_in_sample: inds_in_sample + self.window_size
        ]
        frame_labels = frame_labels[
            inds_in_sample: inds_in_sample + self.window_size
        ]
        if self.transform:
            frames = self.transform(frames)
        if self.target_transform:
            frame_labels = self.target_transform(frame_labels)

        return frames, frame_labels

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
        target_transform: Callable | None = None,
    ):
        """

        Parameters
        ----------
        dataset_path
        window_size
        stride
        split
        transform
        target_transform

        Returns
        -------

        """
        dataset_path = pathlib.Path(dataset_path)
        metadata = Metadata.from_dataset_path(dataset_path)
        frame_dur = metadata.frame_dur

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)

        split_path = dataset_path / split
        sample_ids_path = split_path / constants.SAMPLE_IDS_ARRAY_FILENAME
        sample_ids = np.load(sample_ids_path)
        inds_in_sample_path = (
            split_path / constants.INDS_IN_SAMPLE_ARRAY_FILENAME
        )
        inds_in_sample = np.load(inds_in_sample_path)

        window_inds_path = split_path / constants.WINDOW_INDS_ARRAY_FILENAME
        if window_inds_path.exists():
            window_inds = np.load(window_inds_path)
        else:
            window_inds = None

        return cls(
            dataset_path,
            dataset_df,
            split,
            sample_ids,
            inds_in_sample,
            window_size,
            frame_dur,
            stride,
            window_inds,
            transform,
            target_transform,
        )
