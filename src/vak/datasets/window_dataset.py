from crowsetta import Transcriber
import numpy as np
import pandas as pd
import torch
from torchvision.datasets.vision import VisionDataset

from .. import util


class WindowDataset(VisionDataset):
    """Dataset class that represents all possible windows
     of a fixed width from a set of spectrograms.
     The underlying dataset consists of spectrograms
     of vocalizations and annotations for those vocalizations.

    Returns windows from the spectrograms, along with labels for each
    time bin in the window, derived from the annotations.

    Attributes
    ----------
    root : str, Path
        path to a .csv file that represents the dataset.
        Name 'root' is used for consistency with torchvision.datasets
    x_inds : numpy.ndarray
        indices of each window in the dataset
    spect_id_vector : numpy.ndarray
        represents the 'id' of any spectrogram,
        i.e., the index into spect_paths that will let us load it
    spect_inds_vector : numpy.ndarray
        valid indices of windows we can grab from each spectrogram
    spect_paths : numpy.ndarray
        column from DataFrame that represents dataset,
        consisting of paths to files containing spectrograms as arrays
    annots : list
        of crowsetta.Annotation instances,
        loaded from from DataFrame that represents dataset, using vak.annotation.from_df.
    labelmap : dict
        that maps labels from dataset to a series of consecutive integer.
        To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.
    window_size : int
        number of time bins in windows that will be taken from spectrograms
    spect_key : str
        key to access spectograms in array files. Default is 's'.
    timebins_key : str
        key to access time bin vector in array files. Default is 't'.
    transform : callable
        A function/transform that takes in a numpy array or torch Tensor
        and returns a transformed version. E.g, vak.transforms.StandardizeSpect
        Default is None.
    target_transform : callable
        A function/transform that takes in the target and transforms it.

    Notes
    -----
    This class uses three vectors to represent
    a dataset of windows from spectrograms, without actually loading
    all the spectrograms and concatenating them into one big matrix.
    The three vectors correspond to this imaginary, unloaded big matrix:
    (1) `spect_id_vector` that represents the 'id' of any spectrogram in this matrix,
    i.e., the index into spect_paths that will let us load it, and
    (2) `spect_inds_vector` where the elements represents valid indices of windows
    we can grab from each spectrogram. Valid indices are any up to the index n, where
    n = number of time bins in this spectrogram - number of time bins in our window
    (because if we tried to go past that the window would go past the edge of the
    spectrogram).
    (3) `x_inds` is our 'training set' vector, just a set
    of indices (0, 1, ..., m) where m is the length of vectors (1) and (2).

    When we want to grab a batch of size b of windows, we get b indices from x,
    and then index into vectors (1) and (2) so we know which spectrogram files to
    load, and which windows to grab from each spectrogram
    """
    def __init__(self,
                 root,
                 x_inds,
                 spect_id_vector,
                 spect_inds_vector,
                 spect_paths,
                 annots,
                 labelmap,
                 window_size,
                 spect_key='s',
                 timebins_key='t',
                 transform=None,
                 target_transform=None,
                 ):
        """initialize a WindowDataset instance

        Parameters
        ----------
        root : str, Path
            path to a .csv file that represents the dataset.
            Name 'root' is used for consistency with torchvision.datasets
        x_inds : numpy.ndarray
            indices of each window in the dataset
        spect_id_vector : numpy.ndarray
            represents the 'id' of any spectrogram,
            i.e., the index into spect_paths that will let us load it
        spect_inds_vector : numpy.ndarray
            valid indices of windows we can grab from each spectrogram
        spect_paths : numpy.ndarray
            column from DataFrame that represents dataset,
            consisting of paths to files containing spectrograms as arrays
        annots : list
            of crowsetta.Annotation instances,
            loaded from from DataFrame that represents dataset, using vak.annotation.from_df.
        labelmap : dict
            that maps labels from dataset to a series of consecutive integer.
            To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.
        window_size : int
            number of time bins in windows that will be taken from spectrograms
        spect_key : str
            key to access spectograms in array files. Default is 's'.
        timebins_key : str
            key to access time bin vector in array files. Default is 't'.
        transform : callable
            A function/transform that takes in a numpy array or torch Tensor
            and returns a transformed version. E.g, vak.transforms.StandardizeSpect
            Default is None.
        target_transform : callable
            A function/transform that takes in the target and transforms it.
        """
        super(WindowDataset, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.x_inds = x_inds
        self.spect_id_vector = spect_id_vector
        self.spect_inds_vector = spect_inds_vector
        self.spect_paths = spect_paths
        self.spect_key = spect_key
        self.timebins_key = timebins_key
        self.annots = annots
        self.labelmap = labelmap
        if 'unlabeled' in self.labelmap:
            self.unlabeled_label = self.labelmap['unlabeled']
        else:
            # if there is no "unlabeled label" (e.g., because all segments have labels)
            # just assign dummy value that will end up getting replaced by actual labels by label_timebins()
            self.unlabeled_label = 0
        self.window_size = window_size

        tmp_x_ind = 0
        one_x, _ = self.__getitem__(tmp_x_ind)
        # used by vak functions that need to determine size of window,
        # e.g. when initializing a neural network model
        self.shape = one_x.shape

    def __get_window_labelvec(self, idx):
        """helper function that gets batches of training pairs,
        given indices into dataset

        Parameters
        ----------
        idx : integer
            index into dataset

        Returns
        -------
        window : numpy.ndarray
            window from spectrograms
        labelvec : numpy.ndarray
            vector of labels for each timebin in window from spectrogram
        """
        x_ind = self.x_inds[idx]
        spect_id = self.spect_id_vector[x_ind]
        window_start_ind = self.spect_inds_vector[x_ind]

        spect_path = self.spect_paths[spect_id]
        spect_dict = util.path.array_dict_from_path(spect_path)
        spect = spect_dict[self.spect_key]
        timebins = spect_dict[self.timebins_key]

        annot = self.annots[spect_id]  # "annot id" == spect_id if both were taken from rows of DataFrame
        lbls_int = [self.labelmap[lbl] for lbl in annot.seq.labels]
        lbl_tb = util.labels.label_timebins(lbls_int,
                                            annot.seq.onsets_s,
                                            annot.seq.offsets_s,
                                            timebins,
                                            unlabeled_label=self.unlabeled_label)

        window = spect[:, window_start_ind:window_start_ind + self.window_size]
        labelvec = lbl_tb[window_start_ind:window_start_ind + self.window_size]

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
        return len(self.x_inds)

    @staticmethod
    def n_time_bins_spect(spect_path, spect_key='s'):
        """get number of time bins in a spectrogram,
        given a path to the array file containing that spectrogram

        Parameters
        ----------
        spect_path : str, pathlib.Path
            path to an array file containing a spectrogram and associated arrays.
        spect_key : str
            key to access spectograms in array files. Default is 's'.

        Returns
        -------
        spect.shape[-1], the number of time bins in the spectrogram.
        Assumes spectrogram is a 2-d matrix where rows are frequency bins,
        and columns are time bins.
        """
        spect = util.path.array_dict_from_path(spect_path)[spect_key]
        return spect.shape[-1]

    @classmethod
    def from_csv(cls,
                 csv_path,
                 split,
                 labelmap,
                 window_size,
                 spect_key='s',
                 timebins_key='t',
                 spect_id_vector=None,
                 spect_inds_vector=None,
                 transform=None,
                 target_transform=None):
        """given a path to a csv representing a dataset,
        returns an initialized WindowDataset.

        Parameters
        ----------
        csv_path : str, Path
            path to csv that represents dataset.
        split : str
            name of split from dataset to use
        labelmap : dict
            that maps labels from dataset to a series of consecutive integer.
            To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.
        window_size : int
            number of time bins in windows that will be taken from spectrograms
        spect_key : str
            key to access spectograms in array files. Default is 's'.
        timebins_key : str
            key to access time bin vector in array files. Default is 't'.
        spect_id_vector : numpy.ndarray
            represents the 'id' of any spectrogram,
            i.e., the index into spect_paths that will let us load it
        spect_inds_vector : numpy.ndarray
            valid indices of windows we can grab from each spectrogram
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
        if spect_id_vector is None and spect_inds_vector is not None:
            raise ValueError(
                'must provide spect_id_vector when specifying spect_inds_vector'
            )

        if spect_id_vector is not None and spect_inds_vector is None:
            raise ValueError(
                'must provide spect_inds_vector when specifying spect_id_vector'
            )

        if spect_id_vector is not None and spect_inds_vector is not None:
            if not type(spect_id_vector) is np.ndarray:
                raise TypeError(
                    f'spect_id_vector must be a numpy.ndarray but type was: {type(spect_id_vector)}'
                )

            if not type(spect_inds_vector) is np.ndarray:
                raise TypeError(
                    f'spect_inds_vector must be a numpy.ndarray but type was: {type(spect_inds_vector)}'
                )

            spect_id_vector = util.validation.column_or_1d(spect_id_vector)
            spect_inds_vector = util.validation.column_or_1d(spect_inds_vector)

            if spect_id_vector.shape[-1] != spect_inds_vector.shape[-1]:
                raise ValueError(
                    'spect_id_vector and spect_inds_vector should be same length, but '
                    f'spect_id_vector.shape[-1] is {spect_id_vector.shape[-1]} and '
                    f'spect_inds_vector.shape[-1] is {spect_inds_vector.shape[-1]}.'
                )

        df = pd.read_csv(csv_path)
        if not df['split'].str.contains(split).any():
            raise ValueError(
                f'split {split} not found in dataset in csv: {csv_path}'
            )
        else:
            df = df[df['split'] == split]

        spect_paths = df['spect_path'].values

        if spect_id_vector is None and spect_inds_vector is None:
            # see Notes in class docstring to understand what these vectors do
            spect_id_vector = []
            spect_inds_vector = []
            for ind, spect_path in enumerate(spect_paths):
                n_tb_spect = cls.n_time_bins_spect(spect_path, spect_key)
                # calculate number of windows we can extract from spectrogram of width time_bins
                n_windows = n_tb_spect - window_size
                spect_id_vector.append(np.ones((n_windows,), dtype=np.int64) * ind)
                spect_inds_vector.append(np.arange(n_windows))
            spect_id_vector = np.concatenate(spect_id_vector)
            spect_inds_vector = np.concatenate(spect_inds_vector)

        x_inds = np.arange(spect_id_vector.shape[0])

        annots = util.annotation.from_df(df)

        # note that we set "root" to csv path
        return cls(csv_path,
                   x_inds,
                   spect_id_vector,
                   spect_inds_vector,
                   spect_paths,
                   annots,
                   labelmap,
                   window_size,
                   spect_key,
                   timebins_key,
                   transform,
                   target_transform
                   )
