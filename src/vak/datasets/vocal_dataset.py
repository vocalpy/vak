import pandas as pd

from .. import annotation
from .. import files
from .. import labeled_timebins


class VocalDataset:
    """Base class to represent a dataset of vocalizations.
    It assumes that the dataset consists of spectrograms of
    vocalizations and, optionally, annotations
    for those vocalizations.
    """
    def __init__(self,
                 csv_path,
                 spect_paths,
                 annots=None,
                 labelmap=None,
                 spect_key='s',
                 timebins_key='t',
                 item_transform=None,
                 ):
        """initialize a VocalDataset instance

        Parameters
        ----------
        csv_path : str, Path
            path to a .csv file that represents the dataset.
        spect_paths : numpy.ndarray
            column from DataFrame that represents dataset,
            consisting of paths to files containing spectrograms as arrays
        annots : list
            of crowsetta.Annotation instances,
            loaded from from DataFrame that represents dataset, using vak.annotation.from_df.
            Default is None, in which case no annotation is returned with each item
            in the dataset.
        labelmap : dict
            that maps labels from dataset to a series of consecutive integer.
            To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.
        spect_key : str
            key to access spectograms in array files. Default is 's'.
        timebins_key : str
            key to access time bin vector in array files. Default is 't'.
        item_transform : callable
            A function / transform that takes an input numpy array or torch Tensor,
            and optionally a target array or Tensor, and returns a dictionary.
            This dictionary is the item returned when indexing into the dataset.
            Default is None.
        """
        self.csv_path = csv_path
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
        self.item_transform = item_transform

        tmp_x_ind = 0
        tmp_item = self.__getitem__(tmp_x_ind)
        # used by vak functions that need to determine size of input,
        # e.g. when initializing a neural network model
        self.shape = tmp_item['source'].shape

    def __getitem__(self, idx):
        spect_path = self.spect_paths[idx]
        spect_dict = files.spect.load(spect_path)
        spect = spect_dict[self.spect_key]

        if self.annots is not None:
            timebins = spect_dict[self.timebins_key]

            annot = self.annots[idx]
            lbls_int = [self.labelmap[lbl] for lbl in annot.seq.labels]
            # "lbl_tb": labeled timebins. Target for output of network
            lbl_tb = labeled_timebins.label_timebins(lbls_int,
                                                     annot.seq.onsets_s,
                                                     annot.seq.offsets_s,
                                                     timebins,
                                                     unlabeled_label=self.unlabeled_label)
            item = self.item_transform(spect, lbl_tb, spect_path)
        else:
            item = self.item_transform(spect, spect_path)

        return item

    def __len__(self):
        """number of batches"""
        return len(self.spect_paths)

    @classmethod
    def from_csv(cls, csv_path, split, labelmap,
                 spect_key='s', timebins_key='t', item_transform=None):
        """given a path to a csv representing a dataset,
        returns an initialized VocalDataset.

        Parameters
        ----------
        csv_path : str, Path
            path to a .csv file that represents the dataset.
        spect_paths : numpy.ndarray
            column from DataFrame that represents dataset,
            consisting of paths to files containing spectrograms as arrays
        annots : list
            of crowsetta.Annotation instances,
            loaded from from DataFrame that represents dataset, using vak.annotation.from_df.
            Default is None, in which case no annotation is returned with each item
            in the dataset.
        labelmap : dict
            that maps labels from dataset to a series of consecutive integer.
            To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.
        spect_key : str
            key to access spectograms in array files. Default is 's'.
        timebins_key : str
            key to access time bin vector in array files. Default is 't'.
        item_transform : callable
            A function / transform that takes an input numpy array or torch Tensor,
            and optionally a target array or Tensor, and returns a dictionary.
            This dictionary is the item returned when indexing into the dataset.
            Default is None.

        Returns
        -------
        initialized instance of VocalDataset
        """
        df = pd.read_csv(csv_path)
        if not df['split'].str.contains(split).any():
            raise ValueError(
                f'split {split} not found in dataset in csv: {csv_path}'
            )
        else:
            df = df[df['split'] == split]

        spect_paths = df['spect_path'].values

        # below, annots will be None if no format is specified in the `annot_format` column of the dataframe.
        # this is intended behavior; makes it possible to use same dataset class for prediction
        annots = annotation.from_df(df)

        return cls(csv_path,
                   spect_paths,
                   annots,
                   labelmap,
                   spect_key,
                   timebins_key,
                   item_transform,
                   )
