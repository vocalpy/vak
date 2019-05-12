import random

import numpy as np

from .classes import VocalDataset


def train_test_dur_split_inds(durs,
                              labels,
                              train_dur,
                              test_dur,
                              val_dur=None):
    """split a dataset into training, test, and validation sets of specified durations.
    Given the durations of a set of audio files and the labels for those files,
    this function returns arrays of indices for splitting up the set into training, test,
    and validation sets.

    Parameters
    ----------
    durs : iterable
        of float. Durations of audio files.
    labels : iterable
        of arrays of str or int. Labels for segments (syllables, phonemes, etc.) in audio files.
    labelset : set, list
        set of unique labels for segments in files. Used to verify that each returned array
        of indices will produce a set that contains instances of all labels found in original
        set.
    train_dur : float
        Target duration for training set, in seconds.
    test_dur : float
        Target duration for test set, in seconds.
    val_dur : float
        Target duration for validation set, in seconds. Default is None.
        If None, no indices are returned for validation set.

    Returns
    -------
    train_inds, test_inds, val_inds : numpy.ndarray
        indices to use with some array-like object to produce sets of specified durations
    """
    if len(durs) != len(labels):
        raise ValueError(
            f"length of dur does not equal length of labels"
        )

    total_dur = sum(durs)
    total_target_dur = sum([train_dur,
                             test_dur,
                             val_dur])
    if total_dur < total_target_dur:
        raise ValueError(
            f'Total duration of dataset, {total_dur} seconds, is less than total target duration of '
            f'training, test, and (if specified) validation sets: {total_target_dur}'
        )

    # main loop that gets datasets
    iter = 1
    all_labels_err = ('Did not successfully divide data into training, '
                      'validation, and test sets of sufficient duration '
                      'after 1000 iterations.'
                      ' Try increasing the total size of the data set.')

    while 1:
        dur_inds = list(range(len(durs)))

        train_inds = []
        test_inds = []
        if val_dur:
            val_inds = []
        else:
            val_inds = None

        train_dur = 0
        val_dur = 0
        test_dur = 0

        choice = ['train', 'test']
        if val_dur:
            choice.append('val')

        while 1:
            # pop durations off list and append to randomly-chosen
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
                train_dur += a_spect[1]  # ind 1 is duration
                if train_dur >= train_set_duration:
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

    return train_inds, test_inds, val_inds


def train_test_dur_split(vds,
                         train_dur,
                         test_dur,
                         val_dur=None):
    """split a VocalizationDataset

    Parameters
    ----------
    vds : vak.dataset.VocalizationDataset

    train_dur : float
        total duration of training set, in seconds. Default is None.
    val_dur : float
        total duration of validation set, in seconds. Default is None.
    test_dur : float
        total duration of test set, in seconds. Default is None.


    Returns
    -------
    train, test, val
    """
    durs = [voc.duration for voc in vds.voc_list]
    labels = [voc.annotation.labels for voc in vds.voc_list]
    train_inds, test_inds, val_inds = train_test_dur_split_inds(durs=durs,
                                                                labels=labels,
                                                                train_dur=train_dur,
                                                                test_dur=test_dur,
                                                                val_dur=val_dur)

    train_vds = VocalDataset(voc_list=[vds.voclist[ind] for ind in train_inds])
    test_vds = VocalDataset(voc_list=[vds.voclist[ind] for ind in test_inds])
    if val_inds:
        val_vds = VocalDataset(voc_list=[vds.voclist[ind] for ind in val_inds])
    else:
        val_vds = None

    if val_vds:
        return train_vds, val_vds, test_vds
    else:
        return train_vds, test_vds
