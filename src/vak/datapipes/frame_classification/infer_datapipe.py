"""A datapipe class used for neural network models with the
frame classification task, where the source data consists of audio signals
or spectrograms of varying lengths."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from ...transforms.defaults.frame_classification import InferItemTransform
from . import constants, helper
from .metadata import Metadata

if TYPE_CHECKING:
    from ...transforms import FramesStandardizer


class InferDatapipe:
    """A datapipe class used for
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
    frames_paths : numpy.ndarray
        Paths to npy files containing frames,
        either spectrograms or audio signals
        that are input to the model.
    frame_labels_paths : numpy.ndarray
        Paths to npy files containing vectors
        with a label for each frame.
        The targets for the outputs of the model.
    input_type : str
        The type of input to the neural network model.
        One of {'audio', 'spect'}.
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
    window_size : int
        Size of windows to return;
        number of frames.
    frames_standardizer : vak.transforms.FramesStandardizer, optional
        Transform applied to frames, the input to the neural network model.
        Optional, default is None.
        If supplied, will be used with the transform applied to inputs and targets,
        :class:`vak.transforms.defaults.frame_classification.TrainItemTransform`.
    """

    def __init__(
        self,
        dataset_path: str | pathlib.Path,
        dataset_df: pd.DataFrame,
        input_type: str,
        split: str,
        sample_ids: npt.NDArray,
        inds_in_sample: npt.NDArray,
        frame_dur: float,
        window_size: int,
        frames_standardizer: FramesStandardizer | None = None,
        frames_padval: float = 0.0,
        frame_labels_padval: int = -1,
        return_padding_mask: bool = False,
        subset: str | None = None,
    ):
        """Initialize a new instance of an :class:`InferDatapipe`.

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
        frame_dur: float
            Duration of a frame, i.e., a single sample in audio
            or a single timebin in a spectrogram.
        frames_standardizer : vak.transforms.FramesStandardizer, optional
            Transform applied to frames, the input to the neural network model.
            Optional, default is None.
            If supplied, will be used with the transform applied to inputs and targets,
            :class:`vak.transforms.defaults.frame_classification.InferItemTransform`.
        window_size : int
            Size of windows to return;
            number of frames.
        frames_padval : float
            Value to pad frames with. Added to end of array, the "right side".
            Argument to PadToWindow transform. Default is 0.0.
        frame_labels_padval : int
            Value to pad frame labels vector with. Added to the end of the array.
            Argument to PadToWindow transform. Default is -1.
            Used with ``ignore_index`` argument of :mod:`torch.nn.CrossEntropyLoss`.
        return_padding_mask : bool
            if True, the dictionary returned by ItemTransform classes will include
            a boolean vector to use for cropping back down to size before padding.
            padding_mask has size equal to width of padded array, i.e. original size
            plus padding at the end, and has values of 1 where
            columns in padded are from the original array,
            and values of 0 where columns were added for padding.
        subset : str, optional
            Name of subset to use.
            If specified, this takes precedence over split.
            Subsets are typically taken from the training data
            for use when generating a learning curve.
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
        if split != "predict":
            self.frame_labels_paths = self.dataset_df[
                constants.MULTI_FRAME_LABELS_PATH_COL_NAME
            ].values
        else:
            self.frame_labels_paths = None
        self.sample_ids = sample_ids
        self.inds_in_sample = inds_in_sample
        self.frame_dur = float(frame_dur)
        self.item_transform = InferItemTransform(
            window_size,
            frames_standardizer,
            frames_padval,
            frame_labels_padval,
            return_padding_mask,
        )

    @property
    def duration(self):
        return self.sample_ids.shape[-1] * self.frame_dur

    @property
    def shape(self):
        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        return tmp_item["frames"].shape

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
        frames_path = self.dataset_path / self.frames_paths[idx]
        frames = self._load_frames(frames_path)
        item = {"frames": frames, "frames_path": frames_path}
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
        window_size: int,
        frames_standardizer: FramesStandardizer | None = None,
        frames_padval: float = 0.0,
        frame_labels_padval: int = -1,
        return_padding_mask: bool = False,
        split: str = "val",
        subset: str | None = None,
    ):
        """Make a :class:`InferDatapipe` instance,
        given the path to a frame classification dataset.

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
        frames_standardizer : vak.transforms.FramesStandardizer, optional
            Transform applied to frames, the input to the neural network model.
            Optional, default is None.
            If supplied, will be used with the transform applied to inputs and targets,
            :class:`vak.transforms.defaults.frame_classification.TrainItemTransform`.
        frames_padval : float
            Value to pad frames with. Added to end of array, the "right side".
            Argument to PadToWindow transform. Default is 0.0.
        frame_labels_padval : int
            Value to pad frame labels vector with. Added to the end of the array.
            Argument to PadToWindow transform. Default is -1.
            Used with ``ignore_index`` argument of :mod:`torch.nn.CrossEntropyLoss`.
        return_padding_mask : bool
            if True, the dictionary returned by ItemTransform classes will include
            a boolean vector to use for cropping back down to size before padding.
            padding_mask has size equal to width of padded array, i.e. original size
            plus padding at the end, and has values of 1 where
            columns in padded are from the original array,
            and values of 0 where columns were added for padding.
        split : str
            The name of a split from the dataset,
            one of {'train', 'val', 'test'}.
            Default is "val".
        subset : str, optional
            Name of subset to use.
            If specified, this takes precedence over split.
            Subsets are typically taken from the training data
            for use when generating a learning curve.

        Returns
        -------
        infer_datapipe : InferDatapipe
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

        return cls(
            dataset_path,
            dataset_df,
            input_type,
            split,
            sample_ids,
            inds_in_sample,
            frame_dur,
            window_size,
            frames_standardizer,
            frames_padval,
            frame_labels_padval,
            return_padding_mask,
            subset,
        )
