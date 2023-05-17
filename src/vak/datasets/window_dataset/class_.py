from __future__ import annotations

import pathlib
from typing import Callable

import crowsetta
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torchvision.datasets.vision import VisionDataset

from ... import (
    annotation,
    files,
    transforms,
    validators
)
from ...core.prep.prep_helper import validate_and_get_timebin_dur

from .helper import vectors_from_df


class WindowDataset(VisionDataset):
    """Dataset class that represents all possible time windows
     of a fixed width from a set of spectrograms.

     The underlying dataset consists of spectrogram files
     of vocalizations, and an optional set of annotations
     for those vocalizations.

    Returns windows from the spectrograms.
    When the dataset includes annotations,
    the returned item includes labels for each
    time bin in the window, derived from those annotations.

    This dataset, in combination with functions that crop
    vectors that represent all windows in ``vak.datasets.window_dataset.helper``,
    also enables training on a dataset of a specified duration.

    Attributes
    ----------
    root : str, Path
        Path to a .csv file that represents the dataset.
        Name 'root' is used for consistency with torchvision.datasets.
    source_ids : numpy.ndarray
        Represents the "ID" of any file,
        i.e., the index into ``spect_paths``
        that will let us load that file.
        For a dataset with :math:`m` files,
        this will be an array of length :math:`T`,
        the total number of time bins across all files,
        with elements :math:`i in (0, 1, ..., m - 1)`
        indicating which time bins
        correspond to which file :math:`m_i`:
         :math:`(0, 0, 0, ..., 1, 1, ..., m - 1, m -1)`.
    source_inds : numpy.ndarray
        Same length as ``source_ids`` but values represent
        indices of time bins within each spectrogram.
        For a data set with :math:`T` total time bins across all files,
        where :math:`t_i` indicates the number of time bins
        in each file :math:`m_i`,
        this will look like
        :math:`(0, 1, ..., t_0, 0, 1, ..., t_1, ... t_m)`.
    window_inds : numpy.ndarray
        Starting indices of each valid window in the dataset.
        The value at ``window_inds[0]``
        represents the start index of the first window; using that
        value, we can index into ``source_ids`` to get the path
        of the spectrogram file to load, and
        we can index into ``source_inds``
        to get a window from the audio or spectrogram itself.
        ``window_inds`` will always be strictly shorter than ``source_ids`` and
        ``source_inds``, because the number of valid time bins in
        each file :math:`m_i` will be at most :math:`t_i - \text{window size}`,
        and cropping to a specified duration will remove
        additional time bins.
    source_paths : numpy.ndarray
        Column from DataFrame that represents dataset,
        consisting of paths to files containing spectrograms as arrays.
    annots : list
        List of crowsetta.Annotation instances,
        loaded from DataFrame that represents dataset,
        e.g., by calling ``vak.annotation.from_df``.
    labelmap : dict
        Dict that maps string labels from dataset to consecutive integer.
        To create a label map, pass a set of labels to the
        ``vak.labels.to_map`` function.
    timebin_dur : float
        Duration of a single time bin in spectrograms.
    window_size : int
        Number of time bins in windows that will be taken from spectrograms.
    spect_key : str
        Key to access spectograms in array files. Default is 's'.
    timebins_key : str
        Key to access time bin vector in array files. Default is 't'.
    transform : callable
        Callable that applies pre-processing to each window
        from the dataset, that becomes a sample :math:`x_i`
        in the training set.
        A function/transform that takes in a numpy array or torch Tensor
        and returns a transformed version. E.g, ``vak.transforms.StandardizeSpect``.
        Default is None.
    target_transform : callable
        Callable that applies pre-processing to each "target" :math:`y_i`
        in the training set.

    Notes
    -----
    This class uses three vectors to represent
    a dataset of windows from spectrograms,
    without actually loading all the spectrograms
    and concatenating them into one big array.
    Instead we use three vectors that correspond
    to an imaginary, unloaded array.
    We repeat their definition from the docstring here,
    with a little more context.
    1. ``source_ids``: represents the identity ("id")
    of any source file in this array,
    i.e., the index into ``source_paths`` from the dataset .csv
    that will let us load the source file.
    For a dataset with :math:`m` files,
    this will be an array of length :math:`T`,
    the total number of time bins across all files,
    with elements :math:`i in (0, 1, ..., m - 1)`
    indicating which time bins
    correspond to which file :math:`m_i`:
     :math:`(0, 0, 0, ..., 1, 1, ..., m - 1, m -1)`.
    2. ``source_inds``: Same length as ``source_ids`` but values represent
    indices of time bins within each spectrogram.
    For a data set with :math:`T` total time bins across all files,
    where :math:`t_i` indicates the number of time bins
    in each file :math:`m_i`,
    this will look like
    :math:`(0, 1, ..., t_0, 0, 1, ..., t_1, ... t_m)`.



    where the elements represent valid indices of windows
    we can grab from each spectrogram. Valid indices are any up to the index n, where
    n = number of time bins in this spectrogram - number of time bins in our window
    (because if we tried to go past that the window would go past the edge of the
    spectrogram).
    (3) ``window_inds``: Starting indices of each valid window in the dataset.
    E.g, the value at ``window_inds[0]``
    represents the start index of the first window.
    If a value is in ``window_inds`` it means we can
    we can index into ``source_ids`` at that element and get the path
    of the spectrogram file to load, and then index ``source_inds``
    at the same element to get a *valid* window
    from the spectrogram itself.
    ``window_inds`` will always be strictly shorter than ``source_ids`` and
    ``source_inds``, because the number of valid time bins in
    each file :math:`m_i` will be at most :math:`t_i - \text{window size}`,
    and cropping to a specified duration will remove
    additional time bins.

    When we want to grab a batch of windows of size :math:`b`,
    we get :math:`b` indices from ``window_inds``,
    and then index into ``source_ids`` and ``source_inds``
    so we know which spectrogram files to
    load, and where each window starts
    within each of those spectrograms, respectively.
    In terms of implementation: when a ``torch.DataLoader`` calls ``__getitem__``
    with ``idx``, we index into ``window_inds`` with that ``idx``.
    The number of samples :math:`x_i`
    in a ``WindowDataset`` will be the length :math:`i`
    of ``window_inds``, as returned by `WindowDataset.__len__``.
    """
    # class attribute, constant used by several methods
    # with window_inds, to mark invalid starting indices for windows
    INVALID_WINDOW_VAL = -1

    VALID_SPLITS = ("train", "val" "test", "all")

    def __init__(
        self,
        root: str | pathlib.Path,
        source_ids: npt.NDArray,
        source_inds: npt.NDArray,
        window_inds: npt.NDArray,
        source_paths: list[str | pathlib.Path] | npt.NDArray[np.object],
        annots: list[crowsetta.Annotation],
        labelmap: dict,
        timebin_dur: float,
        window_size: int,
        spect_key: str = "s",
        timebins_key: str = "t",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        r"""Initialize a WindowDataset instance.

        Parameters
        ----------
        root : str, Path
            Path to a .csv file that represents the dataset.
            Name 'root' is used for consistency with torchvision.datasets.
        source_ids : numpy.ndarray
            Represents the "ID" of any file,
            i.e., the index into ``spect_paths``
            that will let us load that file.
            For a dataset with :math:`m` files,
            this will be an array of length :math:`T`,
            the total number of time bins across all files,
            with elements :math:`i in (0, 1, ..., m - 1)`
            indicating which time bins
            correspond to which file :math:`m_i`:
             :math:`(0, 0, 0, ..., 1, 1, ..., m - 1, m -1)`.
        source_inds : numpy.ndarray
            Same length as ``source_ids`` but values represent
            indices of time bins within each spectrogram.
            For a data set with :math:`T` total time bins across all files,
            where :math:`t_i` indicates the number of time bins
            in each file :math:`m_i`,
            this will look like
            :math:`(0, 1, ..., t_0, 0, 1, ..., t_1, ... t_m)`.
        window_inds : numpy.ndarray
            Starting indices of each valid window in the dataset.
            The value at ``window_inds[0]``
            represents the start index of the first window; using that
            value, we can index into ``source_ids`` to get the path
            of the spectrogram file to load, and
            we can index into ``source_inds``
            to get a window from the audio or spectrogram itself.
            ``window_inds`` will always be strictly shorter than ``source_ids`` and
            ``source_inds``, because the number of valid time bins in
            each file :math:`m_i` will be at most :math:`t_i - \text{window size}`,
            and cropping to a specified duration will remove
            additional time bins.
        source_paths : numpy.ndarray
            column from DataFrame that represents dataset,
            consisting of paths to files containing spectrograms as arrays
        annots : list
            List of crowsetta.Annotation instances,
            loaded from DataFrame that represents dataset,
            e.g., by calling ``vak.annotation.from_df``.
        labelmap : dict
            Dict that maps string labels from dataset to consecutive integer.
            To create a label map, pass a set of labels to the
            ``vak.labels.to_map`` function.
        timebin_dur : float
            Duration of a single time bin in spectrograms.
        window_size : int
            Number of time bins in windows that will be taken from spectrograms.
        spect_key : str
            Key to access spectograms in array files. Default is 's'.
        timebins_key : str
            Key to access time bin vector in array files. Default is 't'.
        transform : callable
            Callable that applies pre-processing to each window
            from the dataset, that becomes a sample :math:`x_i`
            in the training set.
            A function/transform that takes in a numpy array or torch Tensor
            and returns a transformed version. E.g, ``vak.transforms.StandardizeSpect``.
            Default is None.
        target_transform : callable
            Callable that applies pre-processing to each "target" :math:`y_i`
            in the training set.
        """
        super(WindowDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.window_inds = window_inds
        self.source_ids = source_ids
        self.source_inds = source_inds
        self.source_paths = source_paths
        self.spect_key = spect_key
        self.timebins_key = timebins_key
        self.annots = annots
        self.labelmap = labelmap
        self.timebin_dur = timebin_dur
        if "unlabeled" in self.labelmap:
            self.unlabeled_label = self.labelmap["unlabeled"]
        else:
            # if there is no "unlabeled" class (e.g., because all segments have labels)
            # just assign dummy value that will end up getting replaced by actual labels by label_timebins()
            self.unlabeled_label = 0
        self.window_size = window_size

        tmp_x_ind = 0
        one_x, _ = self.__getitem__(tmp_x_ind)
        # used by vak functions that need to determine size of window,
        # e.g. when initializing a neural network model
        self.shape = one_x.shape

    def __get_window_labelvec(self, idx: int) -> tuple[npt.NDArray, npt.NDArray]:
        """Helper function that gets batches of training pairs,
        given indices into dataset.

        Parameters
        ----------
        idx : integer
            index into dataset

        Returns
        -------
        window : numpy.ndarray
            Window from spectrogram.
        labelvec : numpy.ndarray
            Vector of labels for each timebin in window from spectrogram.
        """
        x_ind = self.window_inds[idx]
        spect_id = self.source_ids[x_ind]
        window_start_ind = self.source_inds[x_ind]

        spect_path = self.source_paths[spect_id]
        spect_dict = files.spect.load(spect_path)
        spect = spect_dict[self.spect_key]
        timebins = spect_dict[self.timebins_key]

        annot = self.annots[
            spect_id
        ]  # "annot id" == spect_id if both were taken from rows of DataFrame
        lbls_int = [self.labelmap[lbl] for lbl in annot.seq.labels]
        lbl_tb = transforms.labeled_timebins.from_segments(
            lbls_int,
            annot.seq.onsets_s,
            annot.seq.offsets_s,
            timebins,
            unlabeled_label=self.unlabeled_label,
        )

        window = spect[:, window_start_ind : window_start_ind + self.window_size]
        labelvec = lbl_tb[window_start_ind : window_start_ind + self.window_size]

        return window, labelvec

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        window, labelvec = self.__get_window_labelvec(idx)

        if self.transform is not None:
            window = self.transform(window)

        if self.target_transform is not None:
            labelvec = self.target_transform(labelvec)

        return window, labelvec

    def __len__(self):
        """number of batches"""
        return len(self.window_inds)

    @property
    def duration(self):
        """duration of WindowDataset, in seconds"""
        return self.source_inds.shape[-1] * self.timebin_dur

    @classmethod
    def from_csv(
        cls,
        csv_path: str | pathlib.Path,
        split: str,
        labelmap: dict,
        window_size: int,
        crop_dur: float | None = None,
        spect_key: str = "s",
        timebins_key: str = "t",
        source_ids: npt.NDArray | None = None,
        source_inds: npt.NDArray | None = None,
        window_inds: npt.NDArray | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        """Given a path to a csv representing a dataset,
        returns an initialized WindowDataset.

        Parameters
        ----------
        csv_path : str, Path
            Path to csv that represents dataset.
        split : str
            Name of split from dataset to use.
        labelmap : dict
            Dict that maps string labels from dataset to consecutive integer.
            To create a label map, pass a set of labels to the
            ``vak.labels.to_map`` function.
        window_size : int
            Number of time bins in windows that will be taken from spectrograms.
        crop_dur : float
            Duration to which dataset should be "cropped", in seconds.
        spect_key : str
            Key to access spectograms in array files. Default is 's'.
        timebins_key : str
            Key to access time bin vector in array files. Default is 't'.
        source_ids : numpy.ndarray
            Represents the "ID" of any file,
            i.e., the index into ``spect_paths``
            that will let us load that file.
            For a dataset with :math:`m` files,
            this will be an array of length :math:`T`,
            the total number of time bins across all files,
            with elements :math:`i in (0, 1, ..., m - 1)`
            indicating which time bins
            correspond to which file :math:`m_i`:
             :math:`(0, 0, 0, ..., 1, 1, ..., m - 1, m -1)`.
            Default is None, in which case the vector will be
            generated by calling
            ``vak.datasets.window_dataset.helper.from_csv_path``.
        source_inds : numpy.ndarray
            Same length as ``source_ids`` but values represent
            indices of time bins within each spectrogram.
            For a data set with :math:`T` total time bins across all files,
            where :math:`t_i` indicates the number of time bins
            in each file :math:`m_i`,
            this will look like
            :math:`(0, 1, ..., t_0, 0, 1, ..., t_1, ... t_m)`.
            Default is None, in which case the vector will be
            generated by calling
            ``vak.datasets.window_dataset.helper.from_csv_path``.
        window_inds : numpy.ndarray
            Starting indices of each valid window in the dataset.
            The value at ``window_inds[0]``
            represents the start index of the first window; using that
            value, we can index into ``source_ids`` to get the path
            of the spectrogram file to load, and
            we can index into ``source_inds``
            to get a window from the audio or spectrogram itself.
            ``window_inds`` will always be strictly shorter than ``source_ids`` and
            ``source_inds``, because the number of valid time bins in
            each file :math:`m_i` will be at most :math:`t_i - \text{window size}`,
            and cropping to a specified duration will remove
            additional time bins.
            Default is None, in which case the vector will be
            generated by calling
            ``vak.datasets.window_dataset.helper.from_csv_path``.
        transform : callable
            Callable that applies pre-processing to each window
            from the dataset, that becomes a sample :math:`x_i`
            in the training set.
            A function/transform that takes in a numpy array or torch Tensor
            and returns a transformed version. E.g, ``vak.transforms.StandardizeSpect``.
            Default is None.
        target_transform : callable
            Callable that applies pre-processing to each "target" :math:`y_i`
            in the training set.

        source_ids : numpy.ndarray
            represents the 'id' of any spectrogram,
            i.e., the index into source_paths that will let us load it
        source_inds : numpy.ndarray
            valid indices of windows we can grab from each spectrogram
        window_inds : numpy.ndarray
            indices of each window in the dataset. The value at x[0]
            represents the start index of the first window; using that
            value, we can index into source_ids to get the path
            of the spectrogram file to load, and we can index into
            source_inds to index into the spectrogram itself
            and get the window.
        transform : callable
            A function/transform that takes in a numpy array
            and returns a transformed version. E.g, a SpectScaler instance.
            Default is None.
        target_transform : callable
            A function/transform that takes in the target and transforms it.

        Returns
        -------
        initialized instance of WindowDataset
        """
        if any(
            [vec is not None for vec in [source_ids, source_inds, window_inds]]
        ):
            if not all(
                [
                    vec is not None
                    for vec in [source_ids, source_inds, window_inds]
                ]
            ):

                raise ValueError(
                    "if any of the following parameters are specified, they all must be specified: "
                    "source_ids, source_inds, window_inds"
                )

        if all(
            [vec is not None for vec in [source_ids, source_inds, window_inds]]
        ):
            for vec_name, vec in zip(
                ["source_ids", "source_inds", "window_inds"],
                [source_ids, source_inds, window_inds],
            ):
                if not type(vec) is np.ndarray:
                    raise TypeError(
                        f"{vec_name} must be a numpy.ndarray but type was: {type(source_ids)}"
                    )

            source_ids = validators.column_or_1d(source_ids)
            source_inds = validators.column_or_1d(source_inds)
            window_inds = validators.column_or_1d(window_inds)

            if source_ids.shape[-1] != source_inds.shape[-1]:
                raise ValueError(
                    "source_ids and source_inds should be same length, but "
                    f"source_ids.shape[-1] is {source_ids.shape[-1]} and "
                    f"source_inds.shape[-1] is {source_inds.shape[-1]}."
                )

        df = pd.read_csv(csv_path)
        if not df["split"].str.contains(split).any():
            raise ValueError(f"split {split} not found in dataset in csv: {csv_path}")
        else:
            df = df[df["split"] == split]
        source_paths = df["spect_path"].values

        if all([vec is None for vec in [source_ids, source_inds, window_inds]]):
            # see Notes in class docstring to understand what these vectors do
            source_ids, source_inds, window_inds = vectors_from_df(
                df,
                split,
                window_size,
                spect_key=spect_key,
                timebins_key=timebins_key,
                crop_dur = crop_dur,
            )

        annots = annotation.from_df(df)
        timebin_dur = validate_and_get_timebin_dur(df)

        # note that we set "root" to csv path
        return cls(
            csv_path,
            source_ids,
            source_inds,
            window_inds,
            source_paths,
            annots,
            labelmap,
            timebin_dur,
            window_size,
            spect_key,
            timebins_key,
            transform,
            target_transform,
        )
