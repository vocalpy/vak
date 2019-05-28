import copy
import itertools
import os
import random
from glob import glob

import joblib
import numpy as np
from tqdm import tqdm


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
    """converts output of network from label for each frame
    to one label for each continuous segment

    Parameters
    ----------
    labeled_timebins : ndarray
        where each element is a label for a time bin.
        Such an array is the output of the network.
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

    # remove 'unlabeled' label
    if 'unlabeled' in labels_mapping:
        labels = labels[labels != labels_mapping['unlabeled']]
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
        if all([type(el) is str or type(el) is np.str_ for el in labels]):
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
