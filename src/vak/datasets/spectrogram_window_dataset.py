from crowsetta import Transcriber
import numpy as np
import pandas as pd
import torch
from torchvision.datasets.vision import VisionDataset

from .. import util


class SpectrogramWindowDataset(VisionDataset):
    """Dataset class that represents a set of spectrograms of vocalizations
    and annotations for those vocalizations.

    Returns windows from the spectrograms, along with labels for each
    time bin in the window, derived from the annotations."""
    def __init__(self,
                 root,
                 x_inds,
                 spect_id_vector,
                 spect_inds_vector,
                 spect_paths,
                 annot_paths,
                 annot_formats,
                 scribes,
                 labelmap,
                 window_size,
                 spect_key='s',
                 timebins_key='t',
                 transform=None,
                 target_transform=None,
                 ):
        """initialize a SpectrogramWindowDataset instance

        Parameters
        ----------
        root : str, Path
            for SpectrogramWindowDataset, this should be a path
            to a .csv file that represents the dataset.
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
        annot_paths : numpy.ndarray
            column from DataFrame that represents dataset,
            consisting of paths to annotation files
        annot_formats : numpy.ndarray
            column from DataFrame that represents dataset,
            consisting of annotation formats
        scribes : dict
            where keys are unique annotation formats from annot_formats,
            and values are crowsetta.Transcriber instances for those formats
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
            A function/transform that takes in a numpy array
            and returns a transformed version. E.g, a SpectScaler instance.
            Default is None.
        target_transform : callable
            A function/transform that takes in the target and transforms it.
        """
        super(SpectrogramWindowDataset, self).__init__(root, transform=transform,
                                                       target_transform=target_transform)
        self.x_inds = x_inds
        self.spect_id_vector = spect_id_vector
        self.spect_inds_vector = spect_inds_vector
        self.spect_paths = spect_paths
        self.spect_key = spect_key
        self.timebins_key = timebins_key
        self.annot_paths = annot_paths
        self.annot_formats = annot_formats
        self.scribes = scribes
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

        annot_path = self.annot_paths[spect_id]
        annot_format = self.annot_formats[spect_id]
        annot = self.scribes[annot_format].from_file(annot_path)

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

    @classmethod
    def from_csv(cls, csv_path, split, labelmap, window_size,
                 spect_key='s', timebins_key='t', transform=None, target_transform=None):
        """given a path to a csv representing a dataset,
        returns an initialized SpectrogramWindowDataset.

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
        transform : callable
            A function/transform that takes in a numpy array
            and returns a transformed version. E.g, a SpectScaler instance.
            Default is None.
        target_transform : callable
            A function/transform that takes in the target and transforms it.

        Returns
        -------
        initialized instance of SpectrogramWindowDataset
        """
        df = pd.read_csv(csv_path)
        if not df['split'].str.contains(split).any():
            raise ValueError(
                f'split {split} not found in dataset in csv: {csv_path}'
            )
        else:
            df = df[df['split'] == split]

        spect_paths = df['spect_path'].values

        def n_time_bins_spect(spect_path, spect_key=spect_key):
            spect = util.path.array_dict_from_path(spect_path)[spect_key]
            return spect.shape[-1]

        # to represent a dataset of windows from spectrograms without actually loading
        # all the spectrograms and concatenating them into one big matrix,
        # we will make three vectors that correspond to this imaginary, unloaded big matrix:
        # (1) `spect_id_vector` that represents the 'id' of any spectrogram in this matrix,
        # i.e., the index into spect_paths that will let us load it, and
        # (2) `spect_inds_vector` where the elements represents valid indices of windows
        # we can grab from each spectrogram. Valid indices are any up to the index n, where
        # n = number of time bins in this spectrogram - number of time bins in our window
        # (because if we tried to go past that the window would go past the edge of the
        # spectrogram). Lastly we make our 'training set' vector x, which is just a set
        # of indices (0, 1, ..., m) where m is the length of vectors (1) and (2).
        # When we want to grab a batch of size b of windows, we get b indices from x,
        # and then index into vectors (1) and (2) so we know which spectrogram files to
        # load, and which windows to grab from each spectrogram
        spect_id_vector = []  # tells us the index of spect_path
        spect_inds_vector = []  # tells us the index of valid windows in spect loaded from spect_path
        for ind, spect_path in enumerate(spect_paths):
            n_tb_spect = n_time_bins_spect(spect_path)
            # calculate number of windows we can extract from spectrogram of width time_bins
            n_windows = n_tb_spect - window_size
            spect_id_vector.append(np.ones((n_windows,), dtype=np.int64) * ind)
            spect_inds_vector.append(np.arange(n_windows))
        spect_id_vector = np.concatenate(spect_id_vector)
        spect_inds_vector = np.concatenate(spect_inds_vector)

        x_inds = np.arange(spect_id_vector.shape[0])

        annot_paths = df['annot_path'].values
        annot_formats = df['annot_format'].values
        uniq_formats = np.unique(annot_formats)
        scribes = {}
        for annot_format in uniq_formats:
            scribes[annot_format] = Transcriber(annot_format=annot_format)

        # note that we set "root" to csv path
        return cls(csv_path,
                   x_inds,
                   spect_id_vector,
                   spect_inds_vector,
                   spect_paths,
                   annot_paths,
                   annot_formats,
                   scribes,
                   labelmap,
                   window_size,
                   spect_key,
                   timebins_key,
                   transform,
                   target_transform
                   )
