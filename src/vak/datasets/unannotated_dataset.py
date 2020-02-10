import pandas as pd
import torch
from torchvision.datasets.vision import VisionDataset

from .. import util


class UnannotatedDataset(VisionDataset):
    """Dataset class that represents a set of spectrograms
    generated from audio of vocalizations, without annotations.
    Used when predicting annotation.

    Returns (sample, target) pairs where the sample is the spectrogram,
    and the "target" is the path to the file from which the spectrogram was loaded.
    """
    def __init__(self,
                 root,
                 spect_paths,
                 window_size,
                 spect_key='s',
                 timebins_key='t',
                 transform=None,
                 target_transform=None,
                 ):
        """initialize a UnannotatedDataset instance

        Parameters
        ----------
        root : str, Path
            path to a .csv file that represents the dataset.
            Name 'root' is used for consistency with torchvision.datasets
        spect_paths : numpy.ndarray
            column from DataFrame that represents dataset,
            consisting of paths to files containing spectrograms as arrays
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
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.spect_paths = spect_paths
        self.spect_key = spect_key
        self.timebins_key = timebins_key
        self.window_size = window_size

        tmp_x_ind = 0
        one_x, _ = self.__getitem__(tmp_x_ind)
        # used by vak functions that need to determine size of window,
        # e.g. when initializing a neural network model
        self.shape = one_x.shape

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spect_path = self.spect_paths[idx]
        spect_dict = util.path.array_dict_from_path(spect_path)
        spect = spect_dict[self.spect_key]

        if self.transform is not None:
            spect = self.transform(spect)

        if self.target_transform is not None:
            spect_path = self.target_transform(spect_path)

        return spect, spect_path

    def __len__(self):
        """number of batches"""
        return len(self.spect_paths)

    @classmethod
    def from_csv(cls, csv_path, split, window_size,
                 spect_key='s', timebins_key='t',
                 transform=None, target_transform=None):
        """given a path to a csv representing a dataset,
        returns an initialized UnannotatedDataset.

        Parameters
        ----------
        csv_path : str, Path
            path to csv that represents dataset.
        split : str
            name of split from dataset to use
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
        cls: UnannotatedDataset
            initialized instance of UnannotatedDataset
        """
        df = pd.read_csv(csv_path)
        if not df['split'].str.contains(split).any():
            raise ValueError(
                f'split {split} not found in dataset in csv: {csv_path}'
            )
        else:
            df = df[df['split'] == split]

        spect_paths = df['spect_path'].values

        # note that we set "root" to csv path
        return cls(csv_path,
                   spect_paths,
                   window_size,
                   spect_key,
                   timebins_key,
                   transform,
                   target_transform)
