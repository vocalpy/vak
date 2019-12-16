from functools import partial

from crowsetta import Transcriber
import numpy as np
import pandas as pd
import tensorflow as tf

from ... import utils
from ... import io


class BaseDataLoader(tf.keras.utils.Sequence):
    """base DataLoader class used when training neural networks with Keras"""
    def __init__(self, x, y,
                 batch_size,
                 shuffle=True,
                 transform=None,
                 target_transform=None):
        if len(x) != len(y):
            raise ValueError(
                'length of x does not equal length of y'
            )
        self.x = x
        self.y = y
        self.batch_size = batch_size

        self.indices = np.arange(len(x))

        self.shuffle = shuffle
        self.transform = transform
        self.target_transform = target_transform

        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """number of batches"""
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, index):
        """gets one batch"""
        inds = self.indices[index * self.batch_size: (index+1) * self.batch_size]

        # Generate data
        x, y = self.x[inds], self.y[inds]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def on_epoch_end(self):
        """updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    @classmethod
    def from_csv(cls, csv_path, split, batch_size, shuffle=True, transform=None, target_transform=None):
        df = pd.read_csv(csv_path)
        if not df['split'].str.contains(split).any():
            raise ValueError(
                f'split {split} not found in dataset in csv: {csv_path}'
            )
        else:
            df = df[df['split'] == split]

        x = df['spect_files'].values
        y = df['annot_files'].values
        return cls(x, y, batch_size, shuffle, transform, target_transform)


class WindowDataLoader(tf.keras.utils.Sequence):
    """DataLoader class that produces batches of windows from

     used when training neural networks with Keras"""
    def __init__(self,
                 x_inds,
                 spect_id_vector,
                 spect_inds_vector,
                 spect_paths,
                 annot_paths,
                 annot_formats,
                 scribes,
                 labelmap,
                 batch_size,
                 window_size,
                 spect_key='s',
                 timebins_key='t',
                 shuffle=True,
                 spect_scaler=None):
        """

        Parameters
        ----------
        x_inds : numpy.ndarray
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
        batch_size : int
            number of samples in a batch
        window_size : int
            number of time bins in windows that will be taken from spectrograms
        spect_key : str
            key to access spectograms in array files. Default is 's'.
        timebins_key : str
            key to access time bin vector in array files. Default is 't'.
        shuffle : bool
            if True, shuffle dataset. Default is True.
        spect_scaler : vak.utils.spect.SpectScaler
            used to normalize. Default is None.
        """
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

        self.batch_size = batch_size
        self.window_size = window_size

        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.x_inds)

        self.spect_scaler = spect_scaler

        self.to_categorical = partial(tf.keras.utils.to_categorical, num_classes=len(labelmap))

        tmp_x_ind = 0
        one_x, _ = self.__get_x_y(tmp_x_ind)
        self.shape = one_x.shape[1:]

    def __get_x_y(self, x_inds):
        """helper function that gets batches of training pairs,
        given indices into dataset

        Parameters
        ----------
        x_inds : numpy.ndarray
            of indices into dataset

        Returns
        -------
        x : numpy.ndarray
            of windows from spectrograms
        y : numpy.ndarray
            of labels for windows of each timebin
        """
        spect_ids = self.spect_id_vector[x_inds]
        window_start_inds = self.spect_inds_vector[x_inds]

        if np.isscalar(spect_ids) and np.isscalar(window_start_inds):
            # if both are scalar, e.g. as happens when x_inds is scalar,
            # put in a list so we can zip and iterate over them below
            spect_ids = [spect_ids]
            window_start_inds = [window_start_inds]

        x = []
        y = []
        uniq_spect_ids = np.unique(spect_ids)

        for current_spect_id in uniq_spect_ids:
            spect_path = self.spect_paths[current_spect_id]
            spect_dict = io.spect.array_dict_from_path(spect_path)
            spect = spect_dict[self.spect_key]
            timebins = spect_dict[self.timebins_key]

            annot_path = self.annot_paths[current_spect_id]
            annot_format = self.annot_formats[current_spect_id]
            annot = self.scribes[annot_format].from_file(annot_path)

            lbls_int = [self.labelmap[lbl] for lbl in annot.seq.labels]
            lbl_tb = utils.labels.label_timebins(lbls_int,
                                                 annot.seq.onsets_s,
                                                 annot.seq.offsets_s,
                                                 timebins,
                                                 unlabeled_label=self.unlabeled_label)

            these_window_inds = [
                window_ind
                for window_ind, spect_id in zip(window_start_inds, spect_ids)
                if spect_id == current_spect_id
            ]
            for window_ind in these_window_inds:
                x.append(spect[:, window_ind:window_ind + self.window_size])
                y_onehot = self.to_categorical(lbl_tb[window_ind:window_ind + self.window_size])
                y.append(y_onehot)

        if self.spect_scaler:
            x = [self.spect_scaler.transform(window) for window in x]

        x = np.stack(x)
        if x.ndim == 3:  # need to add a "channels" axis
            x = x[:, :, :, np.newaxis]

        y = np.stack(y)

        return x, y

    def __len__(self):
        """number of batches"""
        return int(np.ceil(len(self.x_inds) / float(self.batch_size)))

    def __getitem__(self, index):
        """gets one batch"""
        x_inds = self.x_inds[index * self.batch_size: (index+1) * self.batch_size]
        x, y = self.__get_x_y(x_inds)
        return x, y

    def on_epoch_end(self):
        """updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.x_inds)

    @classmethod
    def from_csv(cls, csv_path, split, labelmap, window_size, batch_size, shuffle=True,
                 spect_key='s', timebins_key='t', spect_scaler=None):
        """given a csv representing a dataset, returns an initialized WindowDataLoader

        Parameters
        ----------
        csv_path : str, Path
            path to csv that represents dataset
        split : str
            name of split from dataset to use
        labelmap : dict
            that maps labels from dataset to a series of consecutive integer.
            To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.
        window_size : int
            number of time bins in windows that will be taken from spectrograms
        batch_size : int
            number of samples in a batch
        shuffle : bool
            if True, shuffle dataset. Default is True.
        spect_key : str
            key to access spectograms in array files. Default is 's'.
        timebins_key : str
            key to access time bin vector in array files. Default is 't'.
        spect_scaler : vak.utils.spect.SpectScaler
            used to normalize. Default is None.

        Returns
        -------
        window_data_loader : WindowDataLoader
            initialized instance of WindowDataLoader
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
            spect = io.spect.array_dict_from_path(spect_path)[spect_key]
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

        return cls(x_inds,
                   spect_id_vector,
                   spect_inds_vector,
                   spect_paths,
                   annot_paths,
                   annot_formats,
                   scribes,
                   labelmap,
                   batch_size,
                   window_size,
                   spect_key=spect_key,
                   timebins_key=timebins_key,
                   shuffle=shuffle,
                   spect_scaler=spect_scaler
                   )
