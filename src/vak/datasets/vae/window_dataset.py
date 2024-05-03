"""Dataset class used for VAE models that operate on fixed-sized windows,
such as a "shotgun VAE" [1]_.

.. [1] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
   Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.
   eLife, 10:e67855. https://doi.org/10.7554/eLife.67855"""
from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..frame_classification import constants, helper
from .metadata import Metadata


class WindowDataset:
    """Dataset class used for VAE models that operate on fixed-sized windows,
    such as a "shotgun VAE" [1]_.

    Attributes
    ----------
    dataset_path : pathlib.Path
        Path to directory that represents a
        frame classification dataset,
        as created by
        :func:`vak.prep.prep_frame_classification_dataset`.
    split : str
        The name of a split from the dataset,
        one of {'train', 'val', 'test'}.
    subset : str, optional
        Name of subset to use.
        If specified, this takes precedence over split.
        Subsets are typically taken from the training data
        for use when generating a learning curve.
    dataset_df : pandas.DataFrame
        A frame classification dataset,
        represented as a :class:`pandas.DataFrame`.
        This will be only the rows that correspond
        to either ``subset`` or ``split`` from the
        ``dataset_df`` that was passed in when
        instantiating the class.
    input_type : str
        The type of input to the neural network model.
        One of {'audio', 'spect'}.
    frame_paths : numpy.ndarray
        Paths to npy files containing frames,
        either spectrograms or audio signals
        that are input to the model.
    sample_ids : numpy.ndarray
        Indexing vector representing which sample
        from the dataset every frame belongs to.
    inds_in_sample : numpy.ndarray
        Indexing vector representing which index
        within each sample from the dataset
        that every frame belongs to.
    window_size : int
        Size of windows to return;
        number of frames.
    frame_dur: float
        Duration of a frame, i.e., a single sample in audio
        or a single timebin in a spectrogram.
    stride : int
        The size of the stride used to determine which windows
        are included in the dataset. The default is 1.
        Used to compute ``window_inds``,
        with the function
        :func:`vak.datasets.frame_classification.window_dataset.get_window_inds`.
    window_inds : numpy.ndarray, optional
        A vector of valid window indices for the dataset.
        If specified, this takes precedence over ``stride``.
    transform : callable
        The transform applied to the frames,
         the input to the neural network :math:`x`.

    References
    ----------
    .. [1] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
       Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.
       eLife, 10:e67855. https://doi.org/10.7554/eLife.67855
    """

    def __init__(
        self,
        dataset_path: str | pathlib.Path,
        dataset_df: pd.DataFrame,
        input_type: str,
        split: str,
        sample_ids: npt.NDArray,
        inds_in_sample: npt.NDArray,
        window_size: int,
        frame_dur: float,
        stride: int = 1,
        subset: str | None = None,
        window_inds: npt.NDArray | None = None,
        transform: Callable | None = None,
    ):
        """Initialize a new instance of a WindowDataset.

        Parameters
        ----------
        dataset_path : pathlib.Path
            Path to directory that represents a
            VAE dataset, as created by :func:`vak.prep.prep_vae_dataset`.
        dataset_df : pandas.DataFrame
            A VAE dataset,
            represented as a :class:`pandas.DataFrame`.
        input_type : str
            The type of input to the neural network model.
            One of {'audio', 'spect'}.
        split : str
            The name of a split from the dataset,
            one of {'train', 'val', 'test'}.
        sample_ids : numpy.ndarray
            Indexing vector representing which sample
            from the dataset every frame belongs to.
        inds_in_sample : numpy.ndarray
            Indexing vector representing which index
            within each sample from the dataset
            that every frame belongs to.
        window_size : int
            Size of windows to return;
            number of frames.
        frame_dur: float
            Duration of a frame, i.e., a single sample in audio
            or a single timebin in a spectrogram.
        stride : int
            The size of the stride used to determine which windows
            are included in the dataset. The default is 1.
            Used to compute ``window_inds``,
            with the function
            :func:`vak.datasets.frame_classification.window_dataset.get_window_inds`.
        subset : str, optional
            Name of subset to use.
            If specified, this takes precedence over split.
            Subsets are typically taken from the training data
            for use when generating a learning curve.
        window_inds : numpy.ndarray, optional
            A vector of valid window indices for the dataset.
            If specified, this takes precedence over ``stride``.
        transform : callable
            The transform applied to the input to the neural network :math:`x`.
        target_transform : callable
            The transform applied to the target for the output
            of the neural network :math:`y`.
        """
        from ... import (
            prep,
        )  # avoid circular import, use for constants.INPUT_TYPES

        if input_type not in prep.constants.INPUT_TYPES:
            raise ValueError(
                f"``input_type`` must be one of: {prep.constants.INPUT_TYPES}\n"
                f"Value for ``input_type`` was: {input_type}"
            )

        self.dataset_path = pathlib.Path(dataset_path)
        self.split = split
        self.subset = subset
        # subset takes precedence over split, if specified
        if subset:
            dataset_df = dataset_df[dataset_df.subset == subset].copy()
        else:
            dataset_df = dataset_df[dataset_df.split == split].copy()
        self.dataset_df = dataset_df
        self.input_type = input_type
        self.frames_paths = self.dataset_df[
            constants.FRAMES_PATH_COL_NAME
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

    def _load_frames(self, frames_path):
        """Helper function that loads "frames",
        the input to the frame classification model.
        Loads audio or spectrogram, depending on
        :attr:`self.input_type`.
        This function assumes that audio is in wav format
        and spectrograms are in npz files.
        """
        return helper.load_frames(frames_path, self.input_type)

    def __getitem__(self, idx):
        window_idx = self.window_inds[idx]
        sample_ids = self.sample_ids[
            window_idx : window_idx + self.window_size  # noqa: E203
        ]
        uniq_sample_ids = np.unique(sample_ids)
        if len(uniq_sample_ids) == 1:
            # we repeat ourselves here to avoid running a loop on one item
            sample_id = uniq_sample_ids[0]
            frames_path = self.dataset_path / self.frames_paths[sample_id]
            frames = self._load_frames(frames_path)

        elif len(uniq_sample_ids) > 1:
            frames = []
            frame_labels = []
            for sample_id in sorted(uniq_sample_ids):
                frames_path = self.dataset_path / self.frames_paths[sample_id]
                frames.append(self._load_frames(frames_path))

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
            ...,
            inds_in_sample : inds_in_sample + self.window_size,  # noqa: E203
        ]
        if self.transform:
            frames = self.transform(frames)

        return frames

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
        subset: str | None = None,
        transform: Callable | None = None,
    ):
        """Make a :class:`WindowDataset` instance,
        given the path to a VAE window dataset.

        Parameters
        ----------
        dataset_path : pathlib.Path
            Path to directory that represents a
            frame classification dataset,
            as created by
            :func:`vak.prep.prep_frame_classification_dataset`.
        window_size : int
            Size of windows to return;
            number of frames.
        stride : int
            The size of the stride used to determine which windows
            are included in the dataset. The default is 1.
            Used to compute ``window_inds``,
            with the function
            :func:`vak.datasets.frame_classification.window_dataset.get_window_inds`.
        split : str
            The name of a split from the dataset,
            one of {'train', 'val', 'test'}.
        subset : str, optional
            Name of subset to use.
            If specified, this takes precedence over split.
            Subsets are typically taken from the training data
            for use when generating a learning curve.
        transform : callable
            The transform applied to the input to the neural network :math:`x`.

        Returns
        -------
        dataset : vak.datasets.vae.WindowDataset
        """
        dataset_path = pathlib.Path(dataset_path)
        metadata = Metadata.from_dataset_path(dataset_path)
        frame_dur = metadata.frame_dur
        input_type = metadata.input_type

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)

        split_path = dataset_path / split
        if subset:
            sample_ids_path = (
                split_path
                / helper.sample_ids_array_filename_for_subset(subset)
            )
        else:
            sample_ids_path = split_path / constants.SAMPLE_IDS_ARRAY_FILENAME
        sample_ids = np.load(sample_ids_path)

        if subset:
            inds_in_sample_path = (
                split_path
                / helper.inds_in_sample_array_filename_for_subset(subset)
            )
        else:
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
            input_type,
            split,
            sample_ids,
            inds_in_sample,
            window_size,
            frame_dur,
            stride,
            subset,
            window_inds,
            transform,
        )
