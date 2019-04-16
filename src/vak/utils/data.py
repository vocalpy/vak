import copy
import itertools
import os
import random
from glob import glob

import joblib
import numpy as np
from scipy.io import wavfile, loadmat
from tqdm import tqdm

from . import spect as spect_utils  # so as not to confuse with variable name `spect`
from vak import evfuncs
from vak.koumura_utils import load_song_annot


def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)


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


def make_spects_from_list_of_files(filelist,
                                   spect_params,
                                   output_dir,
                                   labels_mapping,
                                   skip_files_with_labels_not_in_labelset=True,
                                   annotation_file=None,
                                   n_decimals_trunc=3,
                                   is_for_predict=False):
    """makes spectrograms from a list of audio files

    Parameters
    ----------
    filelist : list
        of str, full paths to .wav or .cbin files
        from which to make spectrograms
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
    annotation_file : str
        full path to file with annotations. Required for .wav files. Expected to
        be a .mat file containing the variables 'keys' and 'elements', where 'keys'
        is a list of .wav filenames and 'elements' are the associated annotations
        for each filename.
        Default is None.
    n_decimals_trunc : int
        number of decimal places to keep when truncating timebin_dur
        default is 3
    is_for_predict : bool
        if True, we are making spectrograms for prediction of labels. This tells function not
        to worry about annotation and to just insert dummy vector of labeled timebins
        into .spect files. Default is False.

    Returns
    -------
    spects_used_path : str
        Full path to file called 'spect_files'
        which contains a list of three-element tuples:
            spect_filename : str, filename of `.spect` file
            spect_dur : float, duration of the spectrogram from cbin
            labels : str, labels from .cbin.not.mat associated with .cbin
                     (string labels for syllables in spectrogram)
        Used when building data sets of a specific duration.

    For each .wav or .cbin filename in the list, a '.spect' file is saved.
    Each '.spect' file contains a "pickled" Python dictionary
    with the following key, value pairs:
        spect : ndarray
            spectrogram
        freq_bins : ndarray
            vector of centers of frequency bins from spectrogram
        time_bins : ndarray
            vector of centers of tme bins from spectrogram
        labeled_timebins : ndarray
            same length as time_bins, but value of each element is a label
            corresponding to that time bin
    """
    decade = 10** n_decimals_trunc  # used below for truncating floating point

    if all(['.cbin' in filename for filename in filelist]):
        filetype = 'cbin'
    elif all(['.wav' in filename for filename in filelist]):
        filetype = 'wav'
    else:
        raise ValueError('Could not determine whether filelist is '
                         'a list of .cbin files or of .wav files. '
                         'All files must be of the same type.')

    if filetype == 'wav':
        if annotation_file is None and is_for_predict is False:
            raise ValueError('annotation_file is required when using .wav files')
        elif annotation_file and is_for_predict is True:
            if annotation_file.endswith('.mat'):
                # the complicated nested structure of the annotation.mat files
                # (a cell array of Matlab structs)
                # makes it hard for loadmat to load them into numpy arrays.
                # Some of the weird looking things below, like accessing fields and
                # then converting them to lists, are work-arounds to deal with
                # the result of loading the complicated structure.
                # Setting squeeze_me=True gets rid of some but not all of the weirdness.
                annotations = loadmat(annotation_file, squeeze_me=True)
                # 'keys' here refers to filenames, which are 'keys' for the 'elements'
                keys_key = [key for key in annotations.keys() if 'keys' in key]
                elements_key = [key for key in annotations.keys() if 'elements' in key]
                if len(keys_key) > 1:
                    raise ValueError('Found more than one `keys` in annotations.mat file')
                if len(elements_key) > 1:
                    raise ValueError('Found more than one `elements` in annotations.mat file')
                if len(keys_key) < 1:
                    raise ValueError('Did not find `keys` in annotations.mat file')
                if len(elements_key) < 1:
                    raise ValueError('Did not find `elements` in annotations.mat file')
                keys_key = keys_key[0]
                elements_key = elements_key[0]
                annot_keys = annotations[keys_key].tolist()
                annot_elements = annotations[elements_key]
            elif annotation_file.endswith('.xml'):
                annotation_dict = load_song_annot(filelist, annotation_file)
        elif annotation_file is None and is_for_predict is True:
            pass
        else:
            ValueError('make_spects_from_list_of_files received annotation file '
                       'but is_for_predict is True')

    # need to keep track of name of files used since we may skip some.
    # (cbins_used is actually a list of tuples as defined in docstring)
    spect_files = []

    pbar = tqdm(filelist)
    for filename in pbar:
        basename = os.path.basename(filename)
        if filetype == 'cbin':
            if not is_for_predict:
                try:
                    notmat_dict = evfuncs.load_notmat(filename)
                except FileNotFoundError:
                    pbar.set_description(
                        f'Did not find .not.mat file for {basename}, skipping file.'
                    )
                    continue
                this_labels_str = notmat_dict['labels']
                onsets = notmat_dict['onsets'] / 1000
                offsets = notmat_dict['offsets'] / 1000
            dat, fs = evfuncs.load_cbin(filename)

        elif filetype == 'wav':
            fs, dat = wavfile.read(filename)
            if not is_for_predict:
                if annotation_file.endswith('.mat'):
                    ind = annot_keys.index(os.path.basename(filename))
                    annotation = annot_elements[ind]
                    # The .tolist() methods calls below are to get the
                    # array out of the weird lengthless object array
                    # that scipy.io.loadmat produces when trying to load
                    # the annotation files.
                    this_labels_str = annotation['segType'].tolist()
                    onsets = annotation['segFileStartTimes'].tolist()
                    offsets = annotation['segFileEndTimes'].tolist()
                elif annotation_file.endswith('.xml'):
                    filename_key = os.path.basename(filename)
                    this_labels_str = annotation_dict[filename_key]['labels']
                    onsets = annotation_dict[filename_key]['onsets'] / fs
                    offsets = annotation_dict[filename_key]['offsets'] / fs

        if not is_for_predict:
            if skip_files_with_labels_not_in_labelset:
                labels_set = set(this_labels_str)
                # below, set(labels_mapping) is a set of that dict's keys
                if not labels_set.issubset(set(labels_mapping)):
                    # because there's some label in labels
                    # that's not in labels_mapping
                    pbar.set_description(
                        f'found labels in {basename} not in labels_mapping, skipping file'
                    )
                    continue

        pbar.set_description(f'making .spect file for {basename}')

        if 'freq_cutoffs' in spect_params:
            dat = spect_utils.butter_bandpass_filter(dat,
                                                     spect_params['freq_cutoffs'][0],
                                                     spect_params['freq_cutoffs'][1],
                                                     fs)

        # have to make a new dictionary with args to spectrogram
        # instead of e.g. using spect_params._asdict()
        # because we don't want spect_params.freq_cutoffs as an arg to spect_utils.spectrogram
        specgram_params = {'fft_size': spect_params.fft_size,
                           'step_size': spect_params.step_size}
        if 'thresh' in spect_params:
            specgram_params['thresh'] = spect_params.thresh
        if 'transform_type' in spect_params:
            specgram_params['transform_type'] = spect_params.transform_type

        spect, freq_bins, time_bins = spect_utils.spectrogram(dat, fs,
                                                              **specgram_params)

        if 'freq_cutoffs' in spect_params:
            f_inds = np.nonzero((freq_bins >= spect_params['freq_cutoffs'][0]) &
                                (freq_bins < spect_params['freq_cutoffs'][1]))[0]  # returns tuple
            spect = spect[f_inds, :]
            freq_bins = freq_bins[f_inds]

        if not is_for_predict:
            this_labels = [labels_mapping[label]
                           for label in this_labels_str]
            this_labeled_timebins = make_labeled_timebins_vector(this_labels,
                                                                 onsets,
                                                                 offsets,
                                                                 time_bins,
                                                                 labels_mapping['silent_gap_label'])
        elif is_for_predict:
            this_labels = []
            this_labels_str = ''
            # below 1 because rows = freq bins, cols = time_bins before we take transpose (as we do in main loop)
            this_labeled_timebins = np.zeros((spect.shape[1], 1))

        if not 'timebin_dur' in locals():
            timebin_dur = np.around(np.mean(np.diff(time_bins)), decimals=3)
        else:
            curr_timebin_dur = np.around(np.mean(np.diff(time_bins)), decimals=3)
            # below truncates any decimal place past decade
            curr_timebin_dur = np.trunc(curr_timebin_dur
                                        * decade) / decade
            if not np.allclose(curr_timebin_dur, timebin_dur):
                raise ValueError("duration of timebin in file {}, {}, did not "
                                 "match duration of timebin from other .mat "
                                 "files, {}.".format(curr_timebin_dur,
                                                     filename,
                                                     timebin_dur))
        spect_dur = time_bins.shape[-1] * timebin_dur

        spect_dict = {'spect': spect,
                      'freq_bins': freq_bins,
                      'time_bins': time_bins,
                      'labels': this_labels_str,
                      'labeled_timebins': this_labeled_timebins,
                      'timebin_dur': timebin_dur,
                      'spect_params': spect_params,
                      'labels_mapping': labels_mapping}

        spect_dict_filename = os.path.join(
            os.path.normpath(output_dir),
            os.path.basename(filename) + '.spect')
        joblib.dump(spect_dict, spect_dict_filename)

        spect_files.append((spect_dict_filename, spect_dur, this_labels_str))

    spect_files_path = os.path.join(output_dir, 'spect_files')
    joblib.dump(spect_files, spect_files_path)

    return spect_files_path


def make_data_dicts(output_dir,
                    total_train_set_duration,
                    validation_set_duration,
                    test_set_duration,
                    labelset,
                    spect_files=None):
    """function that loads data and saves in dictionaries

    Parameters
    ----------
    output_dir : str
        path to output_dir containing .spect files
    total_train_set_duration : float
    validation_set_duration : float
    test_set_duration : float
        all in seconds
    labelset : list
        of str, labels used
    spect_files : str
        full path to file containing 'spect_files' list of tuples
        saved by function make_spects_from_list_of_files.
        Default is None, in which case this function looks for
        a file named 'spect_files' in output_dir.

    Returns
    -------
    saved_data_dict_paths : dict
        with keys {'train', 'test', 'val'} and values being the path
        to which the data_dict was saved

    Saves three 'data_dict' files (train, validation, and test)
    in output_dir, with following structure:
        spects : list
            of ndarray, spectrograms from audio files
        filenames : list
            same length as spects, filename of each audio file that was converted to spectrogram
        freq_bins : ndarray
            vector of frequencies where each value is a bin center. Same for all spectrograms
        time_bins : list
            of ndarrays, each a vector of times where each value is a bin center.
            One for each spectrogram
        labelset : list
            of strings, labels corresponding to each spectrogram
        labeled_timebins : list
            of ndarrays, each same length as time_bins but value is a label for that bin.
            In other words, the labels vector is mapped onto the time_bins vector for the
            spectrogram.
        X : ndarray
            X_train, X_val, or X_test, depending on which data_dict you are looking at.
            Some number of spectrograms concatenated, enough so that the total duration
            of the spectrogram in time bins is equal to or greater than the target duration.
            If greater than target, then X is truncated so it is equal to the target.
        Y : ndarray
            Concatenated labeled_timebins vectors corresponding to the spectrograms in X.
        spect_ID_vector : ndarray
            Vector where each element is an ID for a song. Used to randomly grab subsets
            of data of a target duration while still having the subset be composed of
            individual songs as much as possible. So this vector will look like:
            [0, 0, 0, ..., 1, 1, 1, ... , n, n, n] where n is equal to or (a little) less
            than the length of spects. spect_ID_vector.shape[-1] is the same as X.shape[-1]
            and Y.shape[0].
        timebin_dur : float
            duration of a timebin in seconds from spectrograms
        spect_params : dict
            parameters for computing spectrogram as specified in config.ini file.
            Will be checked against .ini file when running other cli such as learn_curve.py
        labels_mapping : dict
            maps str labels for syllables to consecutive integers.
            As explained in docstring for make_spects_from_list_of_files.
    """
    if not os.path.isdir(output_dir):
        raise NotADirectoryError('{} not recognized '
                                 'as a directory'.format(output_dir))

    if spect_files is None:
        spect_files = glob(os.path.join(output_dir,'spect_files'))
        if spect_files == []:  # if glob didn't find anything
            raise FileNotFoundError("did not find 'spect_files' file in {}"
                                    .format(output_dir))
        elif len(spect_files) > 1:
            raise ValueError("found than more than one 'spect_files' in {}:\n{}"
                             .format(output_dir, spect_files))
        else:
            spect_files = spect_files[0]

    if not os.path.isfile(spect_files):
        raise FileNotFoundError('{} not recognized as a file'
                                .format(spect_files))

    spect_files = joblib.load(spect_files)

    total_spects_dur = sum([spect[1] for spect in spect_files])
    total_dataset_dur = sum([total_train_set_duration,
                             validation_set_duration,
                             test_set_duration])
    if total_spects_dur < total_dataset_dur:
        raise ValueError('Total duration of all .cbin files, {} seconds,'
                         ' is less than total target duration of '
                         'training, validation, and test sets, '
                         '{} seconds'
                         .format(total_spects_dur, total_dataset_dur))

    # main loop that gets datasets
    iter = 1
    all_labels_err = ('Did not successfully divide data into training, '
                      'validation, and test sets of sufficient duration '
                      'after 1000 iterations.'
                      ' Try increasing the total size of the data set.')

    while 1:
        spect_files_copy = copy.deepcopy(spect_files)

        train_spects = []
        val_spects = []
        test_spects = []

        total_train_dur = 0
        val_dur = 0
        test_dur = 0

        choice = ['train', 'val', 'test']

        while 1:
            # pop tuples off cbins_used list and append to randomly-chosen
            # list, either train, val, or test set.
            # Do this until the total duration for each data set is equal
            # to or greater than the target duration for each set.
            try:
                ind = random.randint(0, len(spect_files_copy)-1)
            except ValueError:
                if len(spect_files_copy) == 0:
                    print('Ran out of spectrograms while dividing data into training, '
                          'validation, and test sets of specified durations. Iteration {}'
                          .format(iter))
                    iter += 1
                    break  # do next iteration
                else:
                    raise
            a_spect = spect_files_copy.pop(ind)
            which_set = random.randint(0, len(choice)-1)
            which_set = choice[which_set]
            if which_set == 'train':
                train_spects.append(a_spect)
                total_train_dur += a_spect[1]  # ind 1 is duration
                if total_train_dur >= total_train_set_duration:
                    choice.pop(choice.index('train'))
            elif which_set == 'val':
                val_spects.append(a_spect)
                val_dur += a_spect[1]  # ind 1 is duration
                if val_dur >= validation_set_duration:
                    choice.pop(choice.index('val'))
            elif which_set == 'test':
                test_spects.append(a_spect)
                test_dur += a_spect[1]  # ind 1 is duration
                if test_dur >= test_set_duration:
                    choice.pop(choice.index('test'))

            if len(choice) < 1:
                if np.sum(total_train_dur +
                                  val_dur +
                                  test_dur) < total_dataset_dur:
                    raise ValueError('Loop to find subsets completed but '
                                     'total duration of subsets is less than '
                                     'total duration specified by config file.')
                else:
                    break

            if iter > 1000:
                raise ValueError('Could not find subsets of sufficient duration in '
                                 'less than 1000 iterations.')

        # make sure no contamination between data sets.
        # If this is true, each set of filenames should be disjoint from others
        train_spect_files = [tup[0] for tup in train_spects]  # tup = a tuple
        val_spect_files = [tup[0] for tup in val_spects]
        test_spect_files = [tup[0] for tup in test_spects]
        assert set(train_spect_files).isdisjoint(val_spect_files)
        assert set(train_spect_files).isdisjoint(test_spect_files)
        assert set(val_spect_files).isdisjoint(test_spect_files)

        # make sure that each set contains all classes we
        # want the network to learn
        train_labels = itertools.chain.from_iterable(
            [spect[2] for spect in train_spects])
        train_labels = set(train_labels)  # make set to get unique values

        val_labels = itertools.chain.from_iterable(
            [spect[2] for spect in val_spects])
        val_labels = set(val_labels)

        test_labels = itertools.chain.from_iterable(
            [spect[2] for spect in test_spects])
        test_labels = set(test_labels)

        if train_labels != set(labelset):
            iter += 1
            if iter > 1000:
                raise ValueError(all_labels_err)
            else:
                print('Train labels did not contain all labels in labelset. '
                      'Getting new training set. Iteration {}'
                      .format(iter))
                continue
        elif val_labels != set(labelset):
            iter += 1
            if iter > 1000:
                raise ValueError(all_labels_err)
            else:
                print('Validation labels did not contain all labels in labelset. '
                      'Getting new validation set. Iteration {}'
                      .format(iter))
                continue
        elif test_labels != set(labelset):
            iter += 1
            if iter > 1000:
                raise ValueError(all_labels_err)
            else:
                print('Test labels did not contain all labels in labelset. '
                      'Getting new test set. Iteration {}'
                      .format(iter))
                continue
        else:
            break

    saved_data_dict_paths = {}

    for dict_name, spect_list, target_dur in zip(['train','val','test'],
                                                 [train_spects,val_spects,test_spects],
                                                 [total_train_set_duration,
                                                  validation_set_duration,
                                                  test_set_duration]):

        spects = []
        filenames = []
        all_time_bins = []
        labels = []
        labeled_timebins = []
        spect_ID_vector = []

        for spect_ind, spect_file in enumerate(spect_list):
            spect_dict = joblib.load(spect_file[0])
            spects.append(spect_dict['spect'])
            filenames.append(spect_file[0])
            all_time_bins.append(spect_dict['time_bins'])
            labels.append(spect_dict['labels'])
            labeled_timebins.append(spect_dict['labeled_timebins'])
            spect_ID_vector.extend([spect_ind] * spect_dict['time_bins'].shape[-1])

            if 'freq_bins' in locals():
                assert np.array_equal(spect_dict['freq_bins'], freq_bins)
            else:
                freq_bins = spect_dict['freq_bins']

            if 'labels_mapping' in locals():
                assert spect_dict['labels_mapping'] == labels_mapping
            else:
                labels_mapping = spect_dict['labels_mapping']

            if 'timebin_dur' in locals():
                assert spect_dict['timebin_dur'] == timebin_dur
            else:
                timebin_dur = spect_dict['timebin_dur']

            if 'spect_params' in locals():
                assert spect_dict['spect_params'] == spect_params
            else:
                spect_params = spect_dict['spect_params']

        X = np.concatenate(spects, axis=1)
        Y = np.concatenate(labeled_timebins)
        spect_ID_vector = np.asarray(spect_ID_vector, dtype='int')
        assert X.shape[-1] == Y.shape[0]  # Y has shape (timebins, 1)
        if X.shape[-1] > target_dur / timebin_dur:
            correct_length = np.round(target_dur / timebin_dur).astype(int)
            X = X[:, :correct_length]
            Y = Y[:correct_length, :]
            spect_ID_vector = spect_ID_vector[:correct_length]

        data_dict = {'spects': spects,
                     'filenames': filenames,
                     'freq_bins': freq_bins,
                     'time_bins': all_time_bins,
                     'labels': labels,
                     'labeled_timebins': labeled_timebins,
                     'spect_ID_vector': spect_ID_vector,
                     'X_' + dict_name: X,
                     'Y_' + dict_name: Y,
                     'timebin_dur': timebin_dur,
                     'spect_params': spect_params,
                     'labels_mapping': labels_mapping}

        print('saving data dictionary in {}'.format(output_dir))
        data_dict_path = os.path.join(output_dir, dict_name + '_data_dict')
        joblib.dump(data_dict, data_dict_path)
        saved_data_dict_paths[dict_name] = data_dict_path

    return saved_data_dict_paths


def make_data_dict_from_spect_files(labelset, spect_files='./spect_files',
                                    output_dir='.', dict_name='test'):
    """load data from a list of spect files and save into a dictionary.
    For running a summary of prediction error on a list of files you choose.

    Parameters
    ----------
    labelset : list
        of str, labels used
    spect_files : str
        full path to file containing 'spect_files' list of tuples
        saved by function make_spects_from_list_of_files.
        Default is None, in which case this function looks for
        a file named 'spect_files' in output_dir.
    output_dir : str
        path to output_dir containing .spect files. Default is '.' (current
        working directory).
    dict_name : str
        prefix to filename '_data_dict' in which the dictionary is saved.
        Default is 'test'.

    Returns
    -------
    data_dict_path

    Function saves 'data_dict' file in output_dir, with following structure:
        spects : list
            of ndarray, spectrograms from audio files
        filenames : list
            same length as spects, filename of each audio file that was converted to spectrogram
        freq_bins : ndarray
            vector of frequencies where each value is a bin center. Same for all spectrograms
        time_bins : list
            of ndarrays, each a vector of times where each value is a bin center.
            One for each spectrogram
        labelset : list
            of strings, labels corresponding to each spectrogram
        labeled_timebins : list
            of ndarrays, each same length as time_bins but value is a label for that bin.
            In other words, the labels vector is mapped onto the time_bins vector for the
            spectrogram.
        X : ndarray
            X_train, X_val, or X_test, depending on which data_dict you are looking at.
            Some number of spectrograms concatenated, enough so that the total duration
            of the spectrogram in time bins is equal to or greater than the target duration.
            If greater than target, then X is truncated so it is equal to the target.
        Y : ndarray
            Concatenated labeled_timebins vectors corresponding to the spectrograms in X.
        spect_ID_vector : ndarray
            Vector where each element is an ID for a song. Used to randomly grab subsets
            of data of a target duration while still having the subset be composed of
            individual songs as much as possible. So this vector will look like:
            [0, 0, 0, ..., 1, 1, 1, ... , n, n, n] where n is equal to or (a little) less
            than the length of spects. spect_ID_vector.shape[-1] is the same as X.shape[-1]
            and Y.shape[0].
        timebin_dur : float
            duration of a timebin in seconds from spectrograms
        spect_params : dict
            parameters for computing spectrogram as specified in config.ini file.
            Will be checked against .ini file when running other cli such as learn_curve.py
        labels_mapping : dict
            maps str labels for syllables to consecutive integers.
            As explained in docstring for make_spects_from_list_of_files.
    """
    spect_list = joblib.load(spect_files)
    spect_files = [tup[0] for tup in spect_list]

    labels = itertools.chain.from_iterable(
        [tup[2] for tup in spect_list])
    labels = set(labels)

    if labels != set(labelset):
        raise ValueError(
            'labels from all spect_files are not consistent with '
            'labels in labelset.')

    spects = []
    filenames = []
    all_time_bins = []
    labels = []
    labeled_timebins = []
    spect_ID_vector = []

    for spect_ind, tup in enumerate(spect_list):
        spect_dict = joblib.load(tup[0])
        spects.append(spect_dict['spect'])
        filenames.append(tup[0])
        all_time_bins.append(spect_dict['time_bins'])
        labels.append(spect_dict['labels'])
        labeled_timebins.append(spect_dict['labeled_timebins'])
        spect_ID_vector.extend([spect_ind] * spect_dict['time_bins'].shape[-1])

        if 'freq_bins' in locals():
            assert np.array_equal(spect_dict['freq_bins'], freq_bins)
        else:
            freq_bins = spect_dict['freq_bins']

        if 'labels_mapping' in locals():
            assert spect_dict['labels_mapping'] == labels_mapping
        else:
            labels_mapping = spect_dict['labels_mapping']

        if 'timebin_dur' in locals():
            assert spect_dict['timebin_dur'] == timebin_dur
        else:
            timebin_dur = spect_dict['timebin_dur']

        if 'spect_params' in locals():
            assert spect_dict['spect_params'] == spect_params
        else:
            spect_params = spect_dict['spect_params']

    X = np.concatenate(spects, axis=1)
    Y = np.concatenate(labeled_timebins)
    spect_ID_vector = np.asarray(spect_ID_vector, dtype='int')
    assert X.shape[-1] == Y.shape[0]  # Y has shape (timebins, 1)

    data_dict = {'spects': spects,
                 'filenames': filenames,
                 'freq_bins': freq_bins,
                 'time_bins': all_time_bins,
                 'labels': labels,
                 'labeled_timebins': labeled_timebins,
                 'spect_ID_vector': spect_ID_vector,
                 'X_' + dict_name: X,
                 'Y_' + dict_name: Y,
                 'timebin_dur': timebin_dur,
                 'spect_params': spect_params,
                 'labels_mapping': labels_mapping}

    data_dict_path = os.path.join(output_dir, dict_name + '_data_dict')
    print('saving data dictionary {} in {}'
          .format(data_dict_path, os.path.abspath(output_dir)))
    joblib.dump(data_dict, data_dict_path)
    return data_dict_path


def get_inds_for_dur(spect_ID_vector,
                     labeled_timebins_vector,
                     labels_mapping,
                     target_duration,
                     timebin_dur_in_s=0.001,
                     max_iter=1000,
                     method='incfreq'):
    """Randomly draw a subset of data with a specific duration.
    Draws spectrograms at random and adds to list until total duration of
    all spectrograms => target_duration, and then truncates at target duration.
    Returns vector of indices to select randomly-drawn subset from total dataset.

    Parameters
    ----------
    spect_ID_vector : ndarray
        vector where each element is an ID for a song. Used to randomly grab subsets
        of data of a target duration while still having the subset be composed of
        individual spectrograms as much as possible. So this vector will look like:
        [0, 0, 0, ..., 1, 1, 1, ... , n, n, n] where each n is the ID for one
        spectrogram, i.e. one audio file.
        spect_ID_vector.shape[-1] is the same as spectrograms.shape[-1] and labels.shape[0].
    labeled_timebins_vector : numpy.ndarray
        vector of same length as spect_ID_vector but each element is a label for the timebin,
        i.e. one of the possible classes.
    labels_mapping : dict
        maps string labels to consecutive integers. Used to check that the randomly drawn
        data set contains all classes.
    target_duration : float
        target duration of training set in seconds.
    timebin_dur_in_s : float
        duration of each timebin, i.e. each column in spectrogram, in seconds.
        Default is 0.001 s (1 ms)
    max_iter : int
        number of iterations to try drawing random subset of song that contains
        all classes in labels mapping.
        Default is 1000.
    method : str
        {'rand', 'incfreq'}
        method by which to obtain subset from training set.
        'incfreq' grabs songs at random but starts from the subset
        that includes the least frequently occurring class. Continues
        to grab randomly in order of increasing frequency until all
        classes are present, and then goes back to 'rand' method.
        'rand' grabs songs totally at random.
        Default is 'incfreq'.

    Returns
    -------
    inds_to_use : numpy.ndarray
        numpy array of indices, to index into rows of X_train that belong to chosen subset
        (assumes X_train is one long spectrogram, consisting of all
        training spectrograms concatenated, and each row being one timebin)
    """
    labeled_timebins_vector = np.squeeze(labeled_timebins_vector)

    if labeled_timebins_vector.ndim > 1:
        raise ValueError('number of dimensions of labeled_timebins_vector should be 1 '
                         '(after np.squeeze), but was equal to {}'
                         .format(labeled_timebins_vector.ndim))

    iter = 1
    with tqdm(total=max_iter) as pbar:
        while 1:  # keep iterating until we randomly draw subset that meets our criteria
            pbar.set_description(
                f'Randomly drawing subset of training data, attempt {iter} of {max_iter}'
            )
            if 'inds_to_use' in locals():
                del inds_to_use

            spect_IDs, spect_timebins = np.unique(spect_ID_vector, return_counts=True)

            if iter == 1:
                # sanity check:
                # spect_IDs should always start from 0
                # and go to n-1 where n is # of spectrograms
                assert np.array_equal(spect_IDs, np.arange(spect_IDs.shape[-1]))

            spect_IDs = spect_IDs.tolist()  # because we need to pop off ids for 'incfreq'

            spect_IDs_in_subset = []
            total_dur_in_timebins = 0

            if method == 'incfreq':
                classes, counts = np.unique(labeled_timebins_vector, return_counts=True)
                int_labels_without_int_flag = [val for val in labels_mapping.values()
                                               if type(val) is int]
                if set(classes) != set(int_labels_without_int_flag):
                    raise ValueError('classes in labeled_timebins_vector '
                                     'do not match classes in labels_mapping.')
                freq_rank = np.argsort(counts).tolist()

                # reason for doing it in this Schliemel-the-painter-looking way is that
                # I want to make sure all classes are represented first, but then
                # go back to just grabbing songs completely at random
                while freq_rank:  # is not an empty list yet
                    curr_class = classes[freq_rank.pop(0)]

                    # if curr_class already represented in subset, skip it
                    if 'inds_to_use' in locals():
                        classes_already_in_subset = np.unique(
                            labeled_timebins_vector[inds_to_use])
                        if curr_class in classes_already_in_subset:
                            continue

                    inds_this_class = np.where(labeled_timebins_vector==curr_class)[0]
                    spect_IDs_this_class = np.unique(spect_ID_vector[inds_this_class])
                    # keep only the spect IDs we haven't popped off main list already
                    spect_IDs_this_class = [spect_ID_this_class
                                            for spect_ID_this_class in spect_IDs_this_class
                                            if spect_ID_this_class in spect_IDs]
                    rand_spect_ID = np.random.choice(spect_IDs_this_class)

                    spect_IDs_in_subset.append(rand_spect_ID)
                    spect_IDs.pop(spect_IDs.index(rand_spect_ID))  # so as not to reuse it
                    # below, [0] because np.where returns tuple
                    spect_ID_inds = np.where(spect_ID_vector == rand_spect_ID)[0]
                    if 'inds_to_use' in locals():
                        inds_to_use = np.concatenate((inds_to_use, spect_ID_inds))
                    else:
                        inds_to_use = spect_ID_inds
                    total_dur_in_timebins += spect_timebins[rand_spect_ID]
                    if total_dur_in_timebins * timebin_dur_in_s >= target_duration:
                        # if total_dur greater than target, need to truncate
                        if total_dur_in_timebins * timebin_dur_in_s > target_duration:
                            correct_length = np.round(target_duration / timebin_dur_in_s).astype(int)
                            inds_to_use = inds_to_use[:correct_length]
                        # (if equal to target, don't need to do anything)
                        break

            if method=='rand' or \
                    total_dur_in_timebins * timebin_dur_in_s < target_duration:

                shuffled_spect_IDs = np.random.permutation(spect_IDs)

                for spect_ID in shuffled_spect_IDs:
                    spect_IDs_in_subset.append(spect_ID)
                    # below, [0] because np.where returns tuple
                    spect_ID_inds = np.where(spect_ID_vector == spect_ID)[0]
                    if 'inds_to_use' in locals():
                        inds_to_use = np.concatenate((inds_to_use, spect_ID_inds))
                    else:
                        inds_to_use = spect_ID_inds
                    total_dur_in_timebins += spect_timebins[spect_ID]
                    if total_dur_in_timebins * timebin_dur_in_s >= target_duration:
                        # if total_dur greater than target, need to truncate
                        if total_dur_in_timebins * timebin_dur_in_s > target_duration:
                            correct_length = np.round(target_duration / timebin_dur_in_s).astype(int)
                            inds_to_use = inds_to_use[:correct_length]
                        # (if equal to target, don't need to do anything)
                        break

            if set(np.unique(labeled_timebins_vector[inds_to_use])) != set(labels_mapping.values()):
                pbar.update(1)
                iter += 1

                if iter > max_iter:
                    raise ValueError('Attempted to draw subset of songs at random '
                                     'that contained all classes '
                                     'more than {} times unsuccessfully. Make sure '
                                     'that all classes are present in training data '
                                     'set.'.format(max_iter))
            else:
                return inds_to_use


def reshape_data_for_batching(X, batch_size, time_steps, Y=None):
    """Reshape data to feed to network in batches.
    Data is returned with shape (batch_size, num_batches * time_steps, -1)
    so that a batch can be grabbed at random starting at any index on axis 1
    (that is less than "length of axis 1 minus time steps").
    Pads data with zeros if it cannot be evenly divided into batches of size batch_size.

    Parameters
    ----------
    X : numpy.ndarray
        2-d matrix, concatenated spectrograms.
        Spectrograms are oriented so rows are time bins and columns are frequency bins.
    batch_size : int
        Number of samples in a batch
    time_steps : int
        Number of time steps in each sample, i.e., number of rows
    Y : numpy.ndarray
        vector of labels with shape (X.shape[0], 1). Default is None, for the case when
        you only have spectrograms in X and need to reshape them
        so you can predict labels Y.

    Returns
    -------
    X : numpy.ndarray
        Spectrograms reshaped to have shape (batch_size, num_batches * time_steps, -1).
    Y : numpy.ndarray
        Labels reshaped, will have shape (batch_size, -1). Only returned if Y is provided as an argument.
    num_batches : int
        num_batches = X.shape[0] // batch_size // time_steps (plus 1, if zero-padding was required).
    """
    num_batches = X.shape[0] // batch_size // time_steps
    rows_to_append = ((num_batches + 1) * time_steps * batch_size) - X.shape[0]

    if rows_to_append > 0:
        X = np.concatenate((X, np.zeros((rows_to_append, X.shape[1]))),
                           axis=0)
        if Y is not None:
            Y = np.concatenate((Y, np.zeros((rows_to_append, 1), dtype=int)), axis=0)
        num_batches = num_batches + 1

    X = X.reshape((batch_size, num_batches * time_steps, -1))
    if Y is not None:
        Y = Y.reshape((batch_size, -1))

    if Y is not None:
        return X, Y, num_batches
    else:
        return X, num_batches


def convert_timebins_to_labels(labeled_timebins,
                               labels_mapping,
                               spect_ID_vector=None):
    """converts output of cnn-bilstm from label for each frame
    to one label for each continuous segment

    Parameters
    ----------
    labeled_timebins : ndarray
        where each element is a label for a time bin.
        Such an array is the output of the cnn-bilstm network.
    labels_mapping : dict
        that maps str labels to consecutive integers.
        The mapping is inverted to convert back to str labels.
    spect_ID_vector : ndarray
        of same length as labeled_timebins, where each element
        is an ID # for the spectrogram from which labeled_timebins
        was taken.
        If provided, used to split the converted labels back to
        a list of label str, with one for each spectrogram.
        Default is None, in which case the return value is one long str.

    Returns
    -------
    labels : str or list
        labeled_timebins mapped back to label str.
        If spect_ID_vector was provided, then labels is split into a list of str,
        where each str corresponds to predicted labels for each predicted
        segment in each spectrogram as identified by spect_ID_vector.
    """

    idx = np.diff(labeled_timebins, axis=0).astype(np.bool)
    idx = np.insert(idx, 0, True)

    labels = labeled_timebins[idx]

    # remove silent gap label
    silent_gap_label = labels_mapping['silent_gap_label']
    labels = labels[labels != silent_gap_label]
    labels = labels.tolist()

    inverse_labels_mapping = dict((v, k) for k, v
                                  in labels_mapping.items())
    labels = [inverse_labels_mapping[label] for label in labels]

    if spect_ID_vector:
        labels_list = []
        spect_ID_vector = spect_ID_vector[idx]
        labels_arr = np.asarray(labels)
        # need to split up labels by spect_ID_vector
        # this is probably not the most efficient way:
        spect_IDs = np.unique(spect_ID_vector)

        for spect_ID in spect_IDs:
            these = np.where(spect_ID_vector == spect_ID)
            curr_labels = labels_arr[these].tolist()
            if all([type(el) is str for el in curr_labels]):
                labels_list.append(''.join(curr_labels))
            elif all([type(el) is int for el in curr_labels]):
                labels_list.append(curr_labels)
        return labels_list, spect_ID_vector
    else:
        if all([type(el) is str for el in labels]):
            return ''.join(labels)
        elif all([type(el) is int for el in labels]):
            return labels


def range_str(range_str, sort=True):
    """Generate range of ints from a formatted string,
    then convert range from int to str

    Example:
        >>> range_str('1-4,6,9-11')
        ['1','2','3','4','6','9','10','11']

    Takes a range in form of "a-b" and returns
    a list of numbers between a and b inclusive.
    Also accepts comma separated ranges like "a-b,c-d,f"  which will
    return a list with numbers from a to b, c to d, and f.

    Parameters
    ----------
    range_str : str
        of form 'a-b,c'
        where a hyphen indicates a range
        and a comma separates ranges or single numbers
    sort : bool
        If True, sort output before returning. Default is True.

    Returns
    -------
    list_range : list
        of int, produced by parsing range_str
    """
    # adapted from
    # http://code.activestate.com/recipes/577279-generate-list-of-numbers-from-hyphenated-and-comma/
    s = "".join(range_str.split())  # removes white space
    list_range = []
    for substr in range_str.split(','):
        subrange = substr.split('-')
        if len(subrange) not in [1, 2]:
            raise SyntaxError("unable to parse range {} in labelset {}."
                              .format(subrange, substr))
        list_range.extend(
            [int(subrange[0])]
        ) if len(subrange) == 1 else list_range.extend(
            range(int(subrange[0]), int(subrange[1]) + 1))

    if sort:
        list_range.sort()

    return [str(list_int) for list_int in list_range]