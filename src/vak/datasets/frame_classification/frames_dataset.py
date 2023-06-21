from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.typing as npt

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
    X : numpy.ndarray
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
        X: npt.NDArray,
        Y: npt.NDArray,
        sample_ids: npt.NDArray,
        frame_dur: float,
        item_transform: Callable | None = None,
    ):
        self.X = X
        self.Y = Y
        self.sample_ids = sample_ids
        self.frame_dur = float(frame_dur)
        self.item_transform = item_transform

    @property
    def duration(self):
        return self.X.shape[-1] * self.frame_dur

    @property
    def shape(self):
        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        return tmp_item["source"].shape

    def __getitem__(self, idx):
        is_source_id = self.sample_ids == idx
        x = self.X[..., is_source_id]
        y = self.Y[is_source_id]
        if self.item_transform:
            item = self.item_transform(x, y)
        else:
            item = {
                "source": x,
                "annot": y,
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
        split_path = dataset_path / split
        X_path = split_path / constants.INPUT_ARRAY_FILENAME
        X = np.load(X_path)
        Y_path = split_path / constants.FRAME_LABELS_ARRAY_FILENAME
        Y = np.load(Y_path)
        sample_ids_path = split_path / constants.SOURCE_IDS_ARRAY_FILENAME
        sample_ids = np.load(sample_ids_path)
        metadata = Metadata.from_dataset_path(dataset_path)
        frame_dur = metadata.frame_dur

        return cls(
            X,
            Y,
            sample_ids,
            frame_dur,
            item_transform,
        )
