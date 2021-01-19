import numpy as np
import pandas as pd
import random
import torch
from torchvision.datasets.vision import VisionDataset

from .. import annotation
from .. import files
from .. import io
from .. import labeled_timebins
from .. import validators


class WindowDataset(VisionDataset):
    """Dataset class that represents all possible windows
     of a fixed width from a set of spectrograms.
     The underlying dataset consists of spectrograms
     of vocalizations and annotations for those vocalizations.

    Returns windows from the spectrograms, along with labels for each
    time bin in the window, derived from the annotations.

    Abstraction that enables training on a dataset of a specified duraiton.

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
    timebin_dur : float
        duration of a single time bin in spectrograms.
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

    # class attribute, constant used by several methods
    # with x_inds, to mark invalid starting indices for windows
    INVALID_WINDOW_VAL = -1

    def __init__(self,
                 root,
                 x_inds,
                 spect_id_vector,
                 spect_inds_vector,
                 spect_paths,
                 annots,
                 labelmap,
                 timebin_dur,
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
            indices of each window in the dataset. The value at x[0]
            represents the start index of the first window; using that
            value, we can index into spect_id_vector to get the path
            of the spectrogram file to load, and we can index into
            spect_inds_vector to index into the spectrogram itself
            and get the window.
        spect_id_vector : numpy.ndarray
            represents the 'id' of any spectrogram,
            i.e., the index into spect_paths that will let us load it
        spect_inds_vector : numpy.ndarray
            same length as spect_id_vector but values represent
            indices within each spectrogram.
        spect_paths : numpy.ndarray
            column from DataFrame that represents dataset,
            consisting of paths to files containing spectrograms as arrays
        annots : list
            of crowsetta.Annotation instances,
            loaded from from DataFrame that represents dataset, using vak.annotation.from_df.
        labelmap : dict
            that maps labels from dataset to a series of consecutive integer.
            To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.
        timebin_dur : float
            duration of a single time bin in spectrograms.
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
        self.timebin_dur = timebin_dur
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
        spect_dict = files.spect.load(spect_path)
        spect = spect_dict[self.spect_key]
        timebins = spect_dict[self.timebins_key]

        annot = self.annots[spect_id]  # "annot id" == spect_id if both were taken from rows of DataFrame
        lbls_int = [self.labelmap[lbl] for lbl in annot.seq.labels]
        lbl_tb = labeled_timebins.label_timebins(lbls_int,
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

    def duration(self):
        """duration of WindowDataset, in seconds"""
        return self.spect_inds_vector.shape[-1] * self.timebin_dur

    @staticmethod
    def crop_spect_vectors_keep_classes(lbl_tb,
                                        spect_id_vector,
                                        spect_inds_vector,
                                        x_inds,
                                        crop_dur,
                                        timebin_dur,
                                        labelmap,
                                        window_size):
        """crop spect_id_vector and spect_ind_vector to a target duration
        while making sure that all classes are present in the cropped
        vectors

        Parameters
        ----------
        lbl_tb : numpy.ndarray
            labeled timebins, where labels are from the set of values in labelmap.
        spect_id_vector : numpy.ndarray
            represents the 'id' of any spectrogram,
            i.e., the index into spect_paths that will let us load it
        spect_inds_vector : numpy.ndarray
            same length as spect_id_vector but values represent
            indices within each spectrogram.
        x_inds : numpy.ndarray
            indices of each window in the dataset. The value at x[0]
            represents the start index of the first window; using that
            value, we can index into spect_id_vector to get the path
            of the spectrogram file to load, and we can index into
            spect_inds_vector to index into the spectrogram itself
            and get the window.
        crop_dur : float
            duration to which dataset should be "cropped". Default is None,
            in which case entire duration of specified split will be used.
        timebin_dur : float
            duration of a single time bin in spectrograms. Default is None.
        labelmap : dict
            that maps labels from dataset to a series of consecutive integers.
            To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.

        Returns
        -------
        spect_id_cropped : numpy.ndarray
            spect_id_vector after cropping
        spect_inds_cropped : numpy.ndarray
            spect_inds_vector after cropping
        x_inds_updated : numpy.ndarray
            x_inds_vector with starting indices of windows that are invalid
            after the cropping now set to WindowDataset.INVALID_WINDOW_VAL
            so they will be removed
        """
        lbl_tb = validators.column_or_1d(lbl_tb)
        spect_id_vector = validators.column_or_1d(spect_id_vector)
        spect_inds_vector = validators.column_or_1d(spect_inds_vector)
        x_inds = validators.column_or_1d(x_inds)

        lens = (lbl_tb.shape[-1],
                spect_id_vector.shape[-1],
                spect_inds_vector.shape[-1],
                x_inds.shape[-1])
        uniq_lens = set(lens)
        if len(uniq_lens) != 1:
            raise ValueError(
                'lbl_tb, spect_id_vector, spect_inds_vector, and x_inds should all '
                'have the same length, but did not find one unique length. '
                'Lengths of lbl_tb, spect_id_vector, spect_inds_vector, and x_inds_vector '
                f'were: {lens}'
            )

        cropped_length = np.round(crop_dur / timebin_dur).astype(int)

        if spect_id_vector.shape[-1] == cropped_length:
            return spect_id_vector, spect_inds_vector, x_inds

        elif spect_id_vector.shape[-1] < cropped_length:
            raise ValueError(
                f"arrays have length {spect_id_vector.shape[-1]} "
                f"that is shorter than correct length, {cropped_length}, "
                f"(= target duration {crop_dur} / duration of timebins, {timebin_dur})."
            )

        elif spect_id_vector.shape[-1] > cropped_length:
            classes = np.asarray(
                sorted(list(labelmap.values()))
            )

            # try cropping off the end first
            lbl_tb_cropped = lbl_tb[:cropped_length]

            if np.array_equal(np.unique(lbl_tb_cropped), classes):
                x_inds[cropped_length:] = WindowDataset.INVALID_WINDOW_VAL
                return spect_id_vector[:cropped_length], spect_inds_vector[:cropped_length], x_inds

            # try truncating off the front instead
            lbl_tb_cropped = lbl_tb[-cropped_length:]
            if np.array_equal(np.unique(lbl_tb_cropped), classes):
                # set every index *up to but not including* the first valid window start to "invalid"
                x_inds[:-cropped_length] = WindowDataset.INVALID_WINDOW_VAL
                # also need to 'reset' the indexing so it starts at 0. First find current minimum index value
                min_x_ind = x_inds[x_inds != WindowDataset.INVALID_WINDOW_VAL].min()
                # Then set min x ind to 0, min x ind + 1 to 1, min ind + 2 to 2, ...
                x_inds[x_inds != WindowDataset.INVALID_WINDOW_VAL] = \
                    x_inds[x_inds != WindowDataset.INVALID_WINDOW_VAL] - min_x_ind
                return spect_id_vector[-cropped_length:], spect_inds_vector[-cropped_length:], x_inds

            # try cropping silences
            # This is done by seeking segments > window_size + 2 bins and removing from them
            # When using this option we do not crop the spect vector sizes

            # Ignored data is defined as data that does not appear in any training window.
            # This means that there are 3 distinct cases:
            # 1. Silence in the beginning of a file
            # 2. Silence in the middle of a file
            # 3. Silence at the end of the file
            # (here 'end' means the segment prior to the last window_size bins in the file because
            # Those are not used as startpoints of a training window)

            # assigining WindowDataset.INVALID_WINDOW_VAL to x_inds segments
            # in these 3 cases will cause data to be ignored with
            # durations that depend on wether the segments touch the ends of files
            # because we do not ignore non-silence segments.

            # first identify all silence segments larger than the window duration + 2
            if 'unlabeled' in labelmap:
                unlabeled = labelmap['unlabeled']
            else:
                raise ValueError(
                    "was not able to crop spect vectors to specified duration; "
                    "could not crop from start or end, and there are no unlabeled segments "
                    "that could be used to further crop"
                )
            valid_unlabeled = np.logical_and(lbl_tb == unlabeled, x_inds != WindowDataset.INVALID_WINDOW_VAL)
            unlabeled_diff = np.diff(np.concatenate([[0], valid_unlabeled, [0]]))
            unlabeled_onsets = np.where(unlabeled_diff == 1)[0]
            unlabeled_offsets = np.where(unlabeled_diff == -1)[0]
            unlabeled_durations = unlabeled_offsets - unlabeled_onsets
            N_PAD_BINS = 2
            unlabeled_onsets = unlabeled_onsets[unlabeled_durations >= window_size + N_PAD_BINS]
            unlabeled_offsets = unlabeled_offsets[unlabeled_durations >= window_size + N_PAD_BINS]
            unlabeled_durations = unlabeled_durations[unlabeled_durations >= window_size + N_PAD_BINS]
            # indicate silences in the beginning of files
            border_onsets = np.concatenate([[WindowDataset.INVALID_WINDOW_VAL],
                                            x_inds])[unlabeled_onsets] == WindowDataset.INVALID_WINDOW_VAL
            # indicate silences at the end of files
            border_offsets = np.concatenate([x_inds, [WindowDataset.INVALID_WINDOW_VAL]]
                                            )[unlabeled_offsets + 1] == WindowDataset.INVALID_WINDOW_VAL

            # This is how much data can be ignored from each silence segment without ignoring the end of file windows
            num_potential_ignored_data_bins = unlabeled_durations - (window_size + N_PAD_BINS) + \
                                              window_size * border_onsets

            num_bins_to_crop = len(lbl_tb) - cropped_length
            if sum(num_potential_ignored_data_bins) < num_bins_to_crop:
                # This is how much data can be ignored from each silence segment including the end of file windows
                num_potential_ignored_data_bins = unlabeled_durations - (window_size - N_PAD_BINS) + \
                                                  window_size * (border_onsets + border_offsets)
            else:
                border_offsets[:] = False

            # Second we find a ~random combination to remove
            crop_more = 0
            if sum(num_potential_ignored_data_bins) < num_bins_to_crop:
                # if we will still need to crop more we will do so from non-silence segments
                crop_more = num_bins_to_crop - sum(num_potential_ignored_data_bins) + 1
                num_bins_to_crop = sum(num_potential_ignored_data_bins) - 1

            segment_ind = np.arange(len(num_potential_ignored_data_bins))
            random.shuffle(segment_ind)
            last_ind = np.where(np.cumsum(num_potential_ignored_data_bins[segment_ind]) >= num_bins_to_crop)[0][0]
            bins_to_ignore = np.array([], dtype=int)
            for cnt in range(last_ind):
                if border_onsets[segment_ind[cnt]]:  # remove silences at file onsets
                    bins_to_ignore = np.concatenate([bins_to_ignore,
                                                     np.arange(unlabeled_onsets[segment_ind[cnt]],
                                                               unlabeled_offsets[segment_ind[cnt]] - 1)])
                elif border_offsets[segment_ind[cnt]]:  # remove silences at file offsets
                    bins_to_ignore = np.concatenate([bins_to_ignore,
                                                     np.arange(unlabeled_onsets[segment_ind[cnt]] + 1,
                                                               unlabeled_offsets[segment_ind[cnt]])])
                else:  # remove silences within the files
                    bins_to_ignore = np.concatenate([bins_to_ignore,
                                                     np.arange(unlabeled_onsets[segment_ind[cnt]] + 1,
                                                               unlabeled_offsets[segment_ind[cnt]] - 1)])
            left_to_crop = num_bins_to_crop - sum(num_potential_ignored_data_bins[segment_ind[:last_ind]])-border_onsets[segment_ind[last_ind]]*window_size
            if border_onsets[segment_ind[last_ind]]:
                bins_to_ignore = np.concatenate([bins_to_ignore,
                                             np.arange(unlabeled_onsets[segment_ind[last_ind]],
                                                       unlabeled_onsets[segment_ind[last_ind]] + left_to_crop)])
            elif border_offsets[segment_ind[last_ind]]:
                if left_to_crop < num_potential_ignored_data_bins[segment_ind[last_ind]] - window_size:
                    bins_to_ignore = np.concatenate([bins_to_ignore,
                                             np.arange(unlabeled_onsets[segment_ind[last_ind]] + 1,
                                                       unlabeled_onsets[segment_ind[last_ind]] + left_to_crop)])
                else:
                    bins_to_ignore = np.concatenate([bins_to_ignore,
                                             np.arange(unlabeled_onsets[segment_ind[last_ind]] + 1,
                                                       unlabeled_onsets[segment_ind[last_ind]] + left_to_crop - window_size)])
            else:
                bins_to_ignore = np.concatenate([bins_to_ignore,
                                             np.arange(unlabeled_onsets[segment_ind[last_ind]] + 1,
                                                       unlabeled_onsets[segment_ind[last_ind]] + left_to_crop)])
            
            x_inds[bins_to_ignore] = WindowDataset.INVALID_WINDOW_VAL
        
            # we may still need to crop. Try doing it from the beginning of the dataset
            if crop_more > 0:  # This addition can lead to imprecision but only in cases where we ask for very small datasets
                if crop_more > sum(x_inds != WindowDataset.INVALID_WINDOW_VAL):
                     raise ValueError(
                        "was not able to crop spect vectors to specified duration "
                        "in a way that maintained all classes in dataset"
                        )
                extra_bins = x_inds[x_inds != WindowDataset.INVALID_WINDOW_VAL][:crop_more]
                bins_to_ignore = np.concatenate([bins_to_ignore, extra_bins])
                x_inds[bins_to_ignore] = WindowDataset.INVALID_WINDOW_VAL

            if np.array_equal(np.unique(lbl_tb[np.setdiff1d(np.arange(len(lbl_tb)), bins_to_ignore)]), classes):
                return spect_id_vector, spect_inds_vector, x_inds

        raise ValueError(
                "was not able to crop spect vectors to specified duration "
                "in a way that maintained all classes in dataset"
            )

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
        spect = files.spect.load(spect_path)[spect_key]
        return spect.shape[-1]

    @staticmethod
    def spect_vectors_from_df(df,
                              window_size,
                              spect_key='s',
                              timebins_key='t',
                              crop_dur=None,
                              timebin_dur=None,
                              labelmap=None,
                              ):
        """get spect_id_vector and spect_ind_vector from a dataframe
        that represents a dataset of vocalizations.
        See WindowDataset class docstring for
        detailed explanation of these vectors.

        Parameters
        ----------
        df : pandas.DataFrame
            that represents a dataset of vocalizations.
        window_size : int
            number of time bins in windows that will be taken from spectrograms
        spect_key : str
            key to access spectograms in array files. Default is 's'.
        timebins_key : str
            key to access time bin vector in array files. Default is 't'.
        crop_dur : float
            duration to which dataset should be "cropped". Default is None,
            in which case entire duration of specified split will be used.
        timebin_dur : float
            duration of a single time bin in spectrograms. Default is None.
            Used when "cropping" dataset with crop_dur and required if a
            value is specified for that parameter.
        labelmap : dict
            that maps labels from dataset to a series of consecutive integers.
            To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.
            Used when "cropping" dataset with crop_dur and required if a
            value is specified for that parameter.

        Returns
        -------
        spect_id_vector : numpy.ndarray
            represents the 'id' of any spectrogram,
            i.e., the index into spect_paths that will let us load it
        spect_inds_vector : numpy.ndarray
            valid indices of windows we can grab from each spectrogram
        x_inds_updated : numpy.ndarray
            x_inds_vector with starting indices of windows that are invalid
            after the cropping now set to WindowDataset.INVALID_WINDOW_VAL
            so they will be removed
        """
        if crop_dur is not None and timebin_dur is None:
            raise ValueError(
                'must provide timebin_dur when specifying crop_dur'
            )

        if crop_dur is not None and labelmap is None:
            raise ValueError(
                'must provide labelmap when specifying crop_dur'
            )

        if crop_dur is not None and timebin_dur is not None:
            crop_to_dur = True
            crop_dur = float(crop_dur)
            timebin_dur = float(timebin_dur)
            annots = annotation.from_df(df)
            if 'unlabeled' in labelmap:
                unlabeled_label = labelmap['unlabeled']
            else:
                # if there is no "unlabeled label" (e.g., because all segments have labels)
                # just assign dummy value that will end up getting replaced by actual labels by label_timebins()
                unlabeled_label = 0
        else:
            crop_to_dur = False

        spect_paths = df['spect_path'].values

        spect_id_vector = []
        spect_inds_vector = []
        x_inds = []
        total_tb = 0

        if crop_to_dur:
            lbl_tb = []
            spect_annot_map = annotation.source_annot_map(spect_paths, annots)
            for ind, (spect_path, annot) in enumerate(spect_annot_map.items()):
                spect_dict = files.spect.load(spect_path)
                n_tb_spect = spect_dict[spect_key].shape[-1]

                spect_id_vector.append(np.ones((n_tb_spect,), dtype=np.int64) * ind)
                spect_inds_vector.append(np.arange(n_tb_spect))

                valid_x_inds = np.arange(total_tb, total_tb + n_tb_spect)
                last_valid_window_ind = total_tb + n_tb_spect - window_size
                valid_x_inds[valid_x_inds > last_valid_window_ind] = WindowDataset.INVALID_WINDOW_VAL
                x_inds.append(valid_x_inds)

                total_tb += n_tb_spect

                lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
                timebins = spect_dict[timebins_key]
                lbl_tb.append(labeled_timebins.label_timebins(lbls_int,
                                                              annot.seq.onsets_s,
                                                              annot.seq.offsets_s,
                                                              timebins,
                                                              unlabeled_label=unlabeled_label))

            spect_id_vector = np.concatenate(spect_id_vector)
            spect_inds_vector = np.concatenate(spect_inds_vector)
            lbl_tb = np.concatenate(lbl_tb)
            x_inds = np.concatenate(x_inds)

            (spect_id_vector,
             spect_inds_vector,
             x_inds) = WindowDataset.crop_spect_vectors_keep_classes(lbl_tb,
                                                                     spect_id_vector,
                                                                     spect_inds_vector,
                                                                     x_inds,
                                                                     crop_dur,
                                                                     timebin_dur,
                                                                     labelmap,
                                                                     window_size)

        else:  # crop_to_dur is False
            for ind, spect_path in enumerate(spect_paths):
                n_tb_spect = WindowDataset.n_time_bins_spect(spect_path, spect_key)

                spect_id_vector.append(np.ones((n_tb_spect,), dtype=np.int64) * ind)
                spect_inds_vector.append(np.arange(n_tb_spect))

                valid_x_inds = np.arange(total_tb, total_tb + n_tb_spect)
                last_valid_window_ind = total_tb + n_tb_spect - window_size
                valid_x_inds[valid_x_inds > last_valid_window_ind] = WindowDataset.INVALID_WINDOW_VAL
                x_inds.append(valid_x_inds)

                total_tb += n_tb_spect

            spect_id_vector = np.concatenate(spect_id_vector)
            spect_inds_vector = np.concatenate(spect_inds_vector)
            x_inds = np.concatenate(x_inds)

        x_inds = x_inds[x_inds != WindowDataset.INVALID_WINDOW_VAL]
        return spect_id_vector, spect_inds_vector, x_inds

    @staticmethod
    def spect_vectors_from_csv(csv_path,
                               split,
                               window_size,
                               spect_key='s',
                               timebins_key='t',
                               crop_dur=None,
                               timebin_dur=None,
                               labelmap=None):
        """get spect_id_vector and spect_ind_vector from a
        .csv file that represents a dataset of vocalizations.
        See WindowDataset class docstring for
        detailed explanation of these vectors.

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
        labelmap : dict
            that maps labels from dataset to a series of consecutive integers.
            To create a label map, pass a set of labels to the `vak.utils.labels.to_map` function.
        crop_dur : float
            duration to which dataset should be "cropped". Default is None,
            in which case entire duration of specified split will be used.
        timebin_dur : float
            duration of a single time bin in spectrograms. Default is None.
            Used when "cropping" dataset with crop_dur and required if a
            value is specified for that parameter.

        Returns
        -------
        spect_id_vector : numpy.ndarray
            represents the 'id' of any spectrogram,
            i.e., the index into spect_paths that will let us load it
        spect_inds_vector : numpy.ndarray
            valid indices of windows we can grab from each spectrogram
        x_inds_updated : numpy.ndarray
            x_inds_vector with starting indices of windows that are invalid
            after the cropping now set to WindowDataset.INVALID_WINDOW_VAL
            so they will be removed
        """
        df = pd.read_csv(csv_path)

        if not df['split'].str.contains(split).any():
            raise ValueError(
                f'split {split} not found in dataset in csv: {csv_path}'
            )
        else:
            df = df[df['split'] == split]

        return WindowDataset.spect_vectors_from_df(df,
                                                   window_size,
                                                   spect_key,
                                                   timebins_key,
                                                   crop_dur,
                                                   timebin_dur,
                                                   labelmap)

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
                 x_inds=None,
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
            that maps labels from dataset to a series of consecutive integers.
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
        x_inds : numpy.ndarray
            indices of each window in the dataset. The value at x[0]
            represents the start index of the first window; using that
            value, we can index into spect_id_vector to get the path
            of the spectrogram file to load, and we can index into
            spect_inds_vector to index into the spectrogram itself
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
        if any([vec is not None for vec in [spect_id_vector, spect_inds_vector, x_inds]]):
            if not all([vec is not None for vec in [spect_id_vector, spect_inds_vector, x_inds]]):

                raise ValueError(
                    'if any of the following parameters are specified, they all must be specified: '
                    'spect_id_vector, spect_inds_vector, x_inds'
                )

        if all([vec is not None for vec in [spect_id_vector, spect_inds_vector, x_inds]]):
            for vec_name, vec in zip(['spect_id_vector', 'spect_inds_vector', 'x_inds'],
                                     [spect_id_vector, spect_inds_vector, x_inds]):
                if not type(vec) is np.ndarray:
                    raise TypeError(
                        f'{vec_name} must be a numpy.ndarray but type was: {type(spect_id_vector)}'
                    )

            spect_id_vector = validators.column_or_1d(spect_id_vector)
            spect_inds_vector = validators.column_or_1d(spect_inds_vector)
            x_inds = validators.column_or_1d(x_inds)

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

        if all([vec is None for vec in [spect_id_vector, spect_inds_vector, x_inds]]):
            # see Notes in class docstring to understand what these vectors do
            spect_id_vector, spect_inds_vector, x_inds = cls.spect_vectors_from_df(df, window_size)

        annots = annotation.from_df(df)
        timebin_dur = io.dataframe.validate_and_get_timebin_dur(df)

        # note that we set "root" to csv path
        return cls(csv_path,
                   x_inds,
                   spect_id_vector,
                   spect_inds_vector,
                   spect_paths,
                   annots,
                   labelmap,
                   timebin_dur,
                   window_size,
                   spect_key,
                   timebins_key,
                   transform,
                   target_transform
                   )
