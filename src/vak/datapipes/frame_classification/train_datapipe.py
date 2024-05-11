"""A dataset class used for neural network models with the
frame classification task, where the source data consists of audio signals
or spectrograms of varying lengths.

Unlike :class:`vak.datasets.frame_classification.InferDatapipe`,
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
"""

from __future__ import annotations

import pathlib

import numpy as np
import numpy.typing as npt
import pandas as pd

from ...transforms import FramesStandardizer
from ...transforms.defaults.frame_classification import TrainItemTransform
from . import constants, helper
from .metadata import Metadata


def get_window_inds(n_frames: int, window_size: int, stride: int = 1):
    """Get indices of windows for a :class:`TrainDatapipe`,
    given the number of frames in the dataset,
    the window size, and the stride.

    This function is used by :class:`TrainDatapipe`
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


class TrainDatapipe:
    """Dataset used for training neural network models
    on the frame classification task,
    where the source data consists of audio signals
    or spectrograms of varying lengths.

    Unlike
    :class:`vak.datasets.frame_classification.InferDatapipe`,
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
    :class:`vak.datasets.frame_classification.InferDatapipe`
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
    window_size : int
        Size of windows to return;
        number of frames.
    frames_standardizer : vak.transforms.FramesStandardizer, optional
        Transform applied to frames, the input to the neural network model.
        Optional, default is None.
        If supplied, will be used with the transform applied to inputs and targets,
        :class:`vak.transforms.defaults.frame_classification.TrainItemTransform`.
    frame_dur: float
        Duration of a frame, i.e., a single sample in audio
        or a single timebin in a spectrogram.
    stride : int
        The size of the stride used to determine which windows
        are included in the dataset. The default is 1.
        Used to compute ``window_inds``,
        with the function
        :func:`vak.datasets.frame_classification.train_datapipe.get_window_inds`.
    window_inds : numpy.ndarray, optional
        A vector of valid window indices for the dataset.
        If specified, this takes precedence over ``stride``.
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
        window_size: int,
        frame_dur: float,
        stride: int = 1,
        subset: str | None = None,
        window_inds: npt.NDArray | None = None,
        frames_standardizer: FramesStandardizer | None = None,
    ):
        """Initialize a new instance of a TrainDatapipe.

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
            :func:`vak.datasets.frame_classification.train_datapipe.get_window_inds`.
        subset : str, optional
            Name of subset to use.
            If specified, this takes precedence over split.
            Subsets are typically taken from the training data
            for use when generating a learning curve.
        window_inds : numpy.ndarray, optional
            A vector of valid window indices for the dataset.
            If specified, this takes precedence over ``stride``.
        frames_standardizer : vak.transforms.FramesStandardizer, optional
            Transform applied to frames, the input to the neural network model.
            Optional, default is None.
            If supplied, will be used with the transform applied to inputs and targets,
            :class:`vak.transforms.defaults.frame_classification.TrainItemTransform`.
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
        self.frame_labels_paths = self.dataset_df[
            constants.MULTI_FRAME_LABELS_PATH_COL_NAME
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
        self.item_transform = TrainItemTransform(
            frames_standardizer=frames_standardizer
        )

    @property
    def duration(self):
        return self.sample_ids.shape[-1] * self.frame_dur

    @property
    def shape(self):
        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        # used by vak functions that need to determine size of window,
        # e.g. when initializing a neural network model
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
            frame_labels = np.load(
                self.dataset_path / self.frame_labels_paths[sample_id]
            )

        elif len(uniq_sample_ids) > 1:
            frames = []
            frame_labels = []
            for sample_id in sorted(uniq_sample_ids):
                frames_path = self.dataset_path / self.frames_paths[sample_id]
                frames.append(self._load_frames(frames_path))
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
            ...,
            inds_in_sample : inds_in_sample + self.window_size,  # noqa: E203
        ]
        frame_labels = frame_labels[
            inds_in_sample : inds_in_sample + self.window_size  # noqa: E203
        ]
        item = self.item_transform(frames, frame_labels)
        return item

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
        frames_standardizer: FramesStandardizer | None = None,
    ):
        """Make a :class:`TrainDatapipe` instance,
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
        stride : int
            The size of the stride used to determine which windows
            are included in the dataset. The default is 1.
            Used to compute ``window_inds``,
            with the function
            :func:`vak.datasets.frame_classification.train_datapipe.get_window_inds`.
        split : str
            The name of a split from the dataset,
            one of {'train', 'val', 'test'}.
        subset : str, optional
            Name of subset to use.
            If specified, this takes precedence over split.
            Subsets are typically taken from the training data
            for use when generating a learning curve.
        frames_standardizer : vak.transforms.FramesStandardizer, optional
            Transform applied to frames, the input to the neural network model.
            Optional, default is None.
            If supplied, will be used with the transform applied to inputs and targets,
            :class:`vak.transforms.defaults.frame_classification.TrainItemTransform`.

        Returns
        -------
        dataset : vak.datasets.frame_classification.TrainDatapipe
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
            frames_standardizer,
        )
