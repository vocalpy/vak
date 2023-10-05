"""A dataset class used for neural network models with the
frame classification task, where the source data consists of audio signals
or spectrograms of varying lengths."""
from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import constants, helper
from .metadata import Metadata


class FramesDataset:
    """A dataset class used for
    neural network models
    with the frame classification task,
    where the source data consists of audio signals
    or spectrograms of varying lengths.

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
    sample_ids : numpy.ndarray
        Indexing vector representing which sample
        from the dataset every frame belongs to.
    dataset_df : pandas.DataFrame
        A frame classification dataset,
        represented as a :class:`pandas.DataFrame`.
        This will be only the rows that correspond
        to either ``subset`` or ``split`` from the
        ``dataset_df`` that was passed in when
        instantiating the class.
    frame_paths : numpy.ndarray
        Paths to npy files containing frames,
        either spectrograms or audio signals
        that are input to the model.
    frame_labels_paths : numpy.ndarray
        Paths to npy files containing vectors
        with a label for each frame.
        The targets for the outputs of the model.
    sample_ids : numpy.ndarray
        Indexing vector representing which sample
        from the dataset every frame belongs to.
    inds_in_sample : numpy.ndarray
        Indexing vector representing which index
        within each sample from the dataset
        that every frame belongs to.
    frame_dur: float
        Duration of a frame, i.e., a single sample in audio
        or a single timebin in a spectrogram.
    item_transform : callable, optional
        Transform applied to each item :math:`(x, y)`
        returned by :meth:`FramesDataset.__getitem__`.
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
        subset: str | None = None,
        item_transform: Callable | None = None,
    ):
        """Initialize a new instance of a FramesDataset.

        Parameters
        ----------
        dataset_path : pathlib.Path
            Path to directory that represents a
            frame classification dataset,
            as created by
            :func:`vak.prep.prep_frame_classification_dataset`.
        dataset_df : pandas.DataFrame
            A frame classification dataset,
            represented as a :class:`pandas.DataFrame`.
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
        frame_dur: float
            Duration of a frame, i.e., a single sample in audio
            or a single timebin in a spectrogram.
        input_type : str
            The type of input to the neural network model.
            One of {'audio', 'spect'}.
        subset : str, optional
            Name of subset to use.
            If specified, this takes precedence over split.
            Subsets are typically taken from the training data
            for use when generating a learning curve.
        item_transform : callable, optional
            Transform applied to each item :math:`(x, y)`
            returned by :meth:`FramesDataset.__getitem__`.
        """
        self.dataset_path = pathlib.Path(dataset_path)

        self.split = split
        self.subset = subset
        # subset takes precedence over split, if specified
        if subset:
            dataset_df = dataset_df[dataset_df.subset == subset].copy()
        else:
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
        subset: str | None = None,
        item_transform: Callable | None = None,
    ):
        """

        Parameters
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
        item_transform : callable, optional
            Transform applied to each item :math:`(x, y)`
            returned by :meth:`FramesDataset.__getitem__`.

        Returns
        -------
        frames_dataset : FramesDataset
        """
        dataset_path = pathlib.Path(dataset_path)
        metadata = Metadata.from_dataset_path(dataset_path)
        frame_dur = metadata.frame_dur
        input_type = metadata.input_type

        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)

        split_path = dataset_path / split
        if subset:
            sample_ids_path = split_path / helper.sample_ids_array_filename_for_subset(subset)
        else:
            sample_ids_path = split_path / constants.SAMPLE_IDS_ARRAY_FILENAME
        sample_ids = np.load(sample_ids_path)

        if subset:
            inds_in_sample_path = split_path / helper.inds_in_sample_array_filename_for_subset(subset)
        else:
            inds_in_sample_path = split_path / constants.INDS_IN_SAMPLE_ARRAY_FILENAME
        inds_in_sample = np.load(inds_in_sample_path)

        return cls(
            dataset_path,
            dataset_df,
            split,
            sample_ids,
            inds_in_sample,
            frame_dur,
            input_type,
            subset,
            item_transform,
        )
