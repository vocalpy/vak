import os
from glob import glob
import random
import copy

import numpy as np
import joblib

from . import evfuncs, spect_utils


# adapted from:
# https://github.com/NickleDave/hybrid-vocal-classifier/blob/master/hvc/neuralnet/utils.py
class SpectScaler:
    """class that scales spectrograms that all have the
    same number of frequency bins. Any input spectrogram
    will be scaled by subtracting off the mean of each
    frequency bin from the 'fit' set of spectrograms, and
    then dividing by the standard deviation of each
    frequency bin from the 'fit' set.
    """

    def __init__(self):
        pass

    def fit(self, spect):
        """fit a SpectScaler.
        Input should be spectrogram,
        oriented so that the columns are frequency bins.
        Fit function finds the mean and standard deviation of
        each frequency bin, which are used by `transform` method
        to scale other spectrograms.

        Parameters
        ----------
        spect : 2-d numpy array
            with dimensions (time bins, frequency bins)
        """

        if spect.ndim != 2:
            raise ValueError('input spectrogram should be a 2-d array')

        self.columnMeans = np.mean(spect, axis=0)
        self.columnStds = np.std(spect, axis=0)
        assert self.columnMeans.shape[-1] == spect.shape[-1]
        assert self.columnStds.shape[-1] == spect.shape[-1]
        self.nonZeroStd = np.argwhere(self.columnStds != 0)

    def _transform(self, spect):
        """transforms input spectrogram by subtracting off fit mean
        and then dividing by standard deviation
        """

        transformed = spect - self.columnMeans
        # to keep any zero stds from causing NaNs
        transformed[:, self.nonZeroStd] = (
            transformed[:, self.nonZeroStd] / self.columnStds[self.nonZeroStd])
        return transformed

    def transform(self, spects):
        """normalizes input spectrograms with fit parameters
        Assumes spectrograms are oriented with columns being frequency bins
        and rows being time bins.

        Parameters
        ----------
        spects : 2-d numpy array or list of 2-d numpy arrays
            with dimensions (time bins, frequency bins)

        """

        if any([not hasattr(self, attr) for attr in ['columnMeans',
                                                     'columnStds']]):
            raise AttributeError('SpectScaler properties are set to None,'
                                 'must call fit method first to set the'
                                 'value of these properties before calling'
                                 'transform')

        if type(spects) != np.ndarray and type(spects) != list:
            raise TypeError('type {} is not valid for spects'
                            .format(type(spects)))

        if type(spects) == np.ndarray:
            if spects.shape[-1] != self.columnMeans.shape[-1]:
                raise ValueError('number of columns in spects, {}, '
                                 'does not match shape of self.columnMeans, {},'
                                 'i.e. the number of columns from the spectrogram'
                                 'to which the scaler was fit originally')
            return self._transform(spects)

        elif type(spects) == list:
            z_norm_spects = []
            for spect in spects:
                z_norm_spects.append(self._transform(spect))

            return z_norm_spects

    def fit_transform(self, spects):
        """first calls fit and then returns normalized spects
        transformed using the fit parameters"""

        if type(spects) != np.ndarray:
            raise TypeError('spects passed to fit_transform '
                            'should be numpy array, not {}'
                            .format(type(spects)))

        if spects.ndim != 2:
            raise ValueError('ndims of spects should be 2, not {}'
                             .format(spects.ndim))

        self.fit(spects)
        return self.transform(spects)


def make_labels_mapping_from_dir(data_dir):
    """make mapping
    from the set of unique string labels: [i,a,b,c,d,h,j,k]
    to a sequence of integers: [0,1,2,3,...
    for converting labels into integers
    that can then be converted to one-hot vectors
    for training outputs of a neural network"""
    notmats = glob(data_dir + '*.not.mat')
    labels = []
    for notmat in notmats:
        notmat_dict = evfuncs.load_notmat(notmat)
        label_arr = np.asarray([ord(label)
                                for label in notmat_dict['labels']]
                               )
        labels.append(label_arr)
    labels = np.concatenate(labels)
    uniq_labels = np.unique(labels)
    labels_to_map_to = range(1, uniq_labels.shape[-1] + 1)
    # skip 0 so 0 can be used as label for 'silent gap' across training/testing data
    return dict(zip(uniq_labels, labels_to_map_to))


def make_labeled_timebins_vector(labels,
                                 onsets,
                                 offsets,
                                 time_bins,
                                 silent_gap_label=0):
    """makes a vector of labels for each timebin from a spectrogram,
    given labels for syllables plus onsets and offsets of syllables

    Parameters
    ----------
    labels : ints
        should be mapping returned by make_labels_mapping
    onsets : ndarray
        1d vector of floats, syllable onsets in seconds
    offsets : ndarray
        1d vector of floats, offsets in seconds
    time_bins : ndarray
        1d vector of floats,
        time in seconds for each time bin of a spectrogram
    silent_gap_label : int
        label assigned to silent gaps
        default is 0

    Returns
    -------
    label_vec : ndarray
        same length as time_bins, with each element a label for
        each time bin
    """

    labels = [int(label) for label in labels]
    label_vec = np.ones((time_bins.shape[-1], 1), dtype='int8') * silent_gap_label
    onset_inds = [np.argmin(np.abs(time_bins - onset))
                  for onset in onsets]
    offset_inds = [np.argmin(np.abs(time_bins - offset))
                   for offset in offsets]
    for label, onset, offset in zip(labels, onset_inds, offset_inds):
        label_vec[onset:offset+1] = label
        # offset_inds[ind]+1 because of Matlab one-indexing
    return label_vec


def make_spects_from_list_of_cbins(cbins,
                                   spect_params,
                                   output_dir,
                                   labels_mapping,
                                   skip_files_with_labels_not_in_labelset=True):
    """makes spectrograms from a list of .cbin audio files

    Parameters
    ----------
    cbins : list
         of str, full paths to .cbin files from which to make spectrograms
    spect_params : dict
        parameters for computing spectrogram, from .ini file
    output_dir : str
        directory in which to save .spect file generated for each .cbin file,
        as described below
    labels_mapping : dict
        maps str labels to consecutive integer values {0,1,2,...N} where N
        is the number of classes / label types
    skip_files_with_labels_not_in_labelset : bool
        if True, skip .cbin files where the 'labels' array in the corresponding
        .cbin.not.mat file contains str labels not found in labels_mapping

    Returns
    -------
    cbins_used_path : str
        Full path to cbins_used file
        which contains a list of three-element tuples:
            cbin : str, filename
            spect_dur : float, duration of the spectrogram from cbin
            labels : str, labels from .cbin.not.mat associated with .cbin
                     (string labels for syllables in spectrogram)
        Used when building data sets of a specific duration.

    For each .cbin filename in the list, a "pickled" Python dictionary is saved
    containing the following keys:
        spect : ndarray
            spectrogram
        freq_bins : ndarray
            vector of centers of frequency bins from spectrogram
        time_bins : ndarray
            vector of centers of tme bins from spectrogram
        labeled_timebins : ndarray
            same length as time_bins, but value of each element is a label
            corresponding to that time bin

    Each dictionary is saved with ".spect" appended to the .cbin file name.
    """

    # need to keep track of name of files used since we may skip some.
    # (cbins_used is actually a list of tuples as defined in docstring)
    cbins_used = []

    for cbin in cbins[:number_files]:
        try:
            notmat_dict = evfuncs.load_notmat(cbin)
        except FileNotFoundError:
            print('Did not find .not.mat file for {}, skipping file.'
                  .format(cbin))
            continue

        this_labels = notmat_dict['labels']
        if skip_files_with_labels_not_in_labelset:
            labels_set = set(this_labels)
            # below, set(labels_mapping) is a set of that dict's keys
            if not labels_set.issubset(set(labels_mapping)):
                # because there's some label in labels
                # that's not in labels_mapping
                print('found labels in {} not in labels_mapping, '
                      'skipping file'.format(cbin))
                continue

        dat, fs = evfuncs.load_cbin(cbin)
        if 'freq_cutoffs' in spect_params:
            dat = spect_utils.butter_bandpass_filter(dat,
                                                     spect_params['freq_cutoffs'][0],
                                                     spect_params['freq_cutoffs'][1],
                                                     fs)

        spect, freq_bins, time_bins = spect_utils.spectrogram(dat, fs,
                                                              spect_params['fft_size'],
                                                              spect_params['step_size'],
                                                              spect_params['thresh'],
                                                              spect_params['log_transform'])

        if 'freq_cutoffs' in spect_params:
            f_inds = np.nonzero((freq_bins >= spect_params['freq_cutoffs'][0]) &
                                (freq_bins < spect_params['freq_cutoffs'][1]))[0]  # returns tuple
            spect = spect[f_inds, :]
            freq_bins = freq_bins[f_inds]

        this_labels = [labels_mapping[label]
                  for label in this_labels]
        this_labeled_timebins = make_labeled_timebins_vector(this_labels,
                                                             notmat_dict['onsets'] / 1000,
                                                             notmat_dict['offsets'] / 1000,
                                                             time_bins)

        if not 'timebin_dur' in locals():
            timebin_dur = np.around(np.mean(np.diff(time_bins)), decimals=3)
        else:
            curr_timebin_dur = np.around(np.mean(np.diff(time_bins)), decimals=3)
            assert curr_timebin_dur == timebin_dur, \
                "curr_timebin_dur didn't match timebin_dur"
        spect_dur = time_bins.shape[-1] * timebin_dur

        cbins_used_with_durs.append((cbin, spect_dur, labels))

        data_dict = {'spect': spect,
                     'freq_bins': freq_bins,
                     'time_bins': time_bins,
                     'labels': this_labels,
                     'labeled_timebins': this_labeled_timebins}
        data_dict_filename = cbin + '.spect'
        joblib.dump(data_dict, data_dict_filename)

    cbins_used_path = os.path.join(output_dir, 'cbins_used','w')
    joblib.dump(cbins_used, cbins_used_path)

    return cbins_used_path


def make_data_dicts(output_dir,
                    total_train_set_duration,
                    validation_set_duration,
                    test_set_duration,
                    cbins_used=None):
    """function that loads data and saves in dictionaries

    Parameters
    ----------
    output_dir : str
        path to output_dir containing .spect files
    total_train_set_duration : float
    validation_set_duration : float
    test_set_duration : float
        all in seconds
    cbins_used : str
        full path to file containing 'cbins_used' list of tuples
        saved by function make_spects_from_list_of_cbins.
        Default is None, in which case this function looks for
        a file named 'cbins_used' in output_dir.

    Returns
    -------
    None

    Saves three 'data_dict' files (train, validation, and test)
    in output_dir, with following structure:
        {'X': spects,
         'filenames': cbins_used,
         'freq_bins': freq_bins,
         'time_bins': all_time_bins,
         'Y': labeled_timebins,
         'timebin_dur': timebin_dur}

        where:
            labels : list
                of strings, labels corresponding to each spectrogram
            timebin_dur : float
                duration of a timebin in seconds from spectrograms
            cbins_used : list
                of str, filenames of .cbin files used to generate spects,
                to have a record

    """

    if not os.path.isdir(output_dir):
        raise NotADirectoryError('{} not recognized '
                                 'as a directory'.format(data_dir))

    if cbins_used is None:
        cbins_used = glob(os.path.join(output_dir,'cbins_used'))
        if cbins_used == []:
            raise FileNotFoundError("did not find 'cbins_used' file in {}"
                                    .format(output_dir))


    if not os.path.isfile(cbins_used):
        raise FileNotFoundError('{} not recognized as a file'
                                .format(cbins_used))

    cbins_used = joblib.load(cbins_used)

    total_cbins_dur = sum([cbin[1] for cbin in cbins_used])
    total_dataset_dur = sum([total_train_set_duration,
                             validation_set_duration,
                             test_set_duration])
    if total_cbins_dur < total_dataset_dur:
        raise ValueError('Total duration of all .cbin files, {} seconds,'
                         ' is less than total target duration of '
                         'training, validation, and test sets, '
                         '{} seconds'
                         .format(total_cbins_dur, total_dataset_dur))

    while 1:
        cbins_used_copy = copy.deepcopy(cbins_used)

        train_cbins = []
        val_cbins = []
        test_cbins = []

        total_train_dur = 0
        val_dur = 0
        test_dur = 0

        choice = ['train', 'val', 'test']

        while 1:
            ind = random.randint(len(cbins_used_copy))
            a_cbin = cbins_used_copy.pop(ind)
            which_set = random.randint(len(cbins_used_copy))
            which_set = choice[which_set]
            if which_set == 'train':
                train_cbins.append(a_cbin)
                total_train_dur += a_cbin[1]  # ind 1 is duration
                if total_train_dur > total_train_set_duration:
                    choice.pop(choice.index('train'))
            elif which_set == 'val':
                val_cbins.append(a_cbin)
                val_dur += a_cbin[1]  # ind 1 is duration
                if val_dur > validation_set_duration:
                    choice.pop(choice.index('val'))
            elif which_set == 'test':
                test_cbins.append(a_cbin)
                test_dur += a_cbin[1]  # ind 1 is duration
                if test_dur > test_set_duration:
                    choice.pop(choice.index('test'))

            if len(choice) < 1:
                break

        assert everything
        if everything_works:
            break




    data_dict = {'spects': spects,
                 'filenames': cbins_used,
                 'freq_bins': freq_bins,
                 'time_bins': all_time_bins,
                 'labels': labels,
                 'labeled_timebins': labeled_timebins,
                 'timebin_dur': timebin_dur,
                 'spect_params': spect_params,
                 'labels_mapping': labels_mapping}

    print('saving data dictionary in {}'.format(data_dir))
    data_dict_path = os.path.join(data_dir, 'data_dict')
    joblib.dump(data_dict, data_dict_path)

    return data_dict, data_dict_path


def get_inds_for_dur(song_timebins,
                     target_duration,
                     timebin_dur_in_s=0.001):
    """for getting a training set with random songs but constant duration
    draws songs at random and adds to list
    until total duration of all songs => target_duration
    then truncates at target duration

    Parameters
    ----------
    song_timebins : list
        list of number of timebins for each songfile,
        where timebines is number of rows in a spectrogram
        e.g.,
        [song_spect.shape[0] for song_spect in song_spectrograms]
        (rows are time instead of frequency,
        because network is set up with input this way)
    target_duration : float
        target duration of training set in s
    timebin_dur_in_s : float
        duration of each timebin, i.e. each column in spectrogram,
        in seconds.
        default is 0.001 s (1 ms)

    Returns
    -------
    inds_to_use : bool
        numpy boolean vector, True where row in X_train gets used
        (assumes X_train is one long spectrogram, consisting of all
        training spectrograms concatenated, and each row being one timebin)
    """

    for song_ind, num_timebins_in_song in enumerate(song_timebins):
        inds = np.ones((num_timebins_in_song,), dtype=int) * song_ind
        if 'song_inds_arr' in locals():
            song_inds_arr = np.concatenate((song_inds_arr, inds))
        else:
            song_inds_arr = inds

    song_id_list = []
    total_dur_in_timebins = 0
    num_songs = len(song_timebins)
    while 1:
        song_id = random.randrange(num_songs)
        if song_id in song_id_list:
            continue
        else:
            song_id_list.append(song_id)
            song_id_inds = np.where(song_inds_arr == song_id)[0]  # 0 because where np.returns tuple
            if 'inds_to_use' in locals():
                inds_to_use = np.concatenate((inds_to_use, song_id_inds))
            else:
                inds_to_use = song_id_inds
            total_dur_in_timebins = total_dur_in_timebins + song_timebins[song_id]
            if total_dur_in_timebins * timebin_dur_in_s >= target_duration:
                # if total_dur greater than target, need to truncate
                if total_dur_in_timebins * timebin_dur_in_s > target_duration:
                    correct_length = np.round(target_duration / timebin_dur_in_s).astype(int)
                    inds_to_use = inds_to_use[:correct_length]
                # (if equal to target, don't need to do anything)
                break

    return inds_to_use


def reshape_data_for_batching(X, Y, batch_size, time_steps, input_vec_size):
    """reshape to feed to network in batches"""
    # need to loop through train data in chunks, can't fit on GPU all at once
    # First zero pad
    num_batches = X.shape[0] // batch_size // time_steps
    rows_to_append = ((num_batches + 1) * time_steps * batch_size) - X.shape[0]
    X = np.concatenate((X, np.zeros((rows_to_append, input_vec_size))),
                       axis=0)
    Y = np.concatenate((Y, np.zeros((rows_to_append, 1), dtype=int)), axis=0)
    num_batches = num_batches + 1
    X = X.reshape((batch_size, num_batches * time_steps, -1))
    Y = Y.reshape((batch_size, -1))
    return X, Y, num_batches


def levenshtein(source, target):
    """levenshtein distance
    from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """

    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


def syllable_error_rate(true, pred):
    """syllable error rate: word error rate, but with songbird syllables
    Levenshtein/edit distance normalized by length of true sequence

    Parameters:
    """