import random
import itertools
import warnings

from .classes import VocalizationDataset


def train_test_dur_split_inds(durs,
                              labels,
                              labelset,
                              train_dur=None,
                              test_dur=None,
                              val_dur=None):
    """return indices to split a dataset into training, test, and validation sets of specified durations.

    Given the durations of a set of vocalizations, and labels from the annotations for those vocalizations,
    this function returns arrays of indices for splitting up the set into training, test,
    and validation sets.

    Using those indices will produce datasets that each contain instances of all labels in the set of labels.

    Parameters
    ----------
    durs : iterable
        of float. Durations of audio files.
    labels : iterable
        of numpy arrays of str or int. Labels for segments (syllables, phonemes, etc.) in audio files.
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
            f"length of durs, {len(durs)} does not equal length of labels, {len(labels)}"
        )

    total_dur = sum(durs)

    if train_dur is None and test_dur is None:
        raise ValueError(
            'must specify either train_dur or test_dur'
        )

    if val_dur is None:
        if train_dur is None:
            train_dur = total_dur - test_dur
        elif test_dur is None:
            test_dur = total_dur - train_dur
        total_target_dur = sum([train_dur,
                                test_dur])
    elif val_dur is not None:
        if train_dur is None:
            train_dur = total_dur - (test_dur + val_dur)
        elif test_dur is None:
            test_dur = total_dur - (train_dur + val_dur)
        total_target_dur = sum([train_dur,
                                test_dur,
                                val_dur])

    if total_target_dur < total_dur:
        warnings.warn(
            'Total target duration of training, test, and (if specified) validation sets, '
            f'{total_target_dur} seconds, is less than total duration of dataset: {total_dur:.3f}. '
            'Not all of dataset will be used.'
        )

    if total_target_dur > total_dur:
        raise ValueError(
            f'Total duration of dataset, {total_dur} seconds, is less than total target duration of '
            f'training, test, and (if specified) validation sets: {total_target_dur}'
        )

    iter = 1
    all_labels_err = ('Did not successfully divide data into training, '
                      'validation, and test sets of sufficient duration '
                      'after 1000 iterations.'
                      ' Try increasing the total size of the data set.')

    durs_labels_list = list(zip(durs, labels))

    while 1:
        train_inds = []
        test_inds = []
        if val_dur:
            val_inds = []
        else:
            val_inds = None

        total_train_dur = 0
        total_val_dur = 0
        total_test_dur = 0

        choice = ['train', 'test']
        if val_dur:
            choice.append('val')

        durs_labels_inds = list(range(len(durs_labels_list)))
        random.shuffle(durs_labels_inds)
        while 1:
            # pop durations off list and append to randomly-chosen
            # list, either train, val, or test set.
            # Do this until the total duration for each data set is equal
            # to or greater than the target duration for each set.
            try:
                ind = durs_labels_inds.pop()
            except IndexError:
                if len(durs_labels_inds) == 0:
                    print(
                        'Ran out of elements while dividing dataset into subsets of specified durations.'
                        f'Iteration {iter}'
                          )
                    iter += 1
                    break  # do next iteration
                else:
                    raise

            which_set = random.randint(0, len(choice)-1)
            which_set = choice[which_set]
            if which_set == 'train':
                train_inds.append(ind)
                total_train_dur += durs_labels_list[ind][0]  # ind 0 is duration
                if total_train_dur >= train_dur:
                    choice.pop(choice.index('train'))
            elif which_set == 'val':
                val_inds.append(ind)
                total_val_dur += durs_labels_list[ind][0]  # ind 0 is duration
                if total_val_dur >= val_dur:
                    choice.pop(choice.index('val'))
            elif which_set == 'test':
                test_inds.append(ind)
                total_test_dur += durs_labels_list[ind][0]  # ind 0 is duration
                if total_test_dur >= test_dur:
                    choice.pop(choice.index('test'))

            if len(choice) < 1:
                total_all_durs = total_train_dur + total_test_dur
                if val_dur:
                    total_all_durs += total_val_dur
                if total_all_durs < total_target_dur:
                    raise ValueError(
                        f'Loop to find subsets completed but total duration of subsets, {total_all_durs} seconds, '
                        f'is less than total duration specified: {total_target_dur} seconds.')
                else:
                    break

            if iter > 1000:
                raise ValueError('Could not find subsets of sufficient duration in '
                                 'less than 1000 iterations.')

        # make sure that each set contains all classes we
        # want the network to learn
        train_tups = [durs_labels_list[ind] for ind in train_inds]  # tup = a tuple
        test_tups = [durs_labels_list[ind] for ind in test_inds]
        if val_dur:
            val_tups = [durs_labels_list[ind] for ind in val_inds]

        train_labels = itertools.chain.from_iterable(
            [tup[1] for tup in train_tups])
        train_labels = set(train_labels)  # make set to get unique values

        test_labels = itertools.chain.from_iterable(
            [tup[1] for tup in test_tups])
        test_labels = set(test_labels)

        if val_dur:
            val_labels = itertools.chain.from_iterable(
                [tup[1] for tup in val_tups])
            val_labels = set(val_labels)

        if train_labels != set(labelset):
            iter += 1
            if iter > 1000:
                raise ValueError(all_labels_err)
            else:
                print('Train labels did not contain all labels in labelset. '
                      'Getting new training set. Iteration {}'
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
        elif val_dur is not None and val_labels != set(labelset):
            iter += 1
            if iter > 1000:
                raise ValueError(all_labels_err)
            else:
                print('Validation labels did not contain all labels in labelset. '
                      'Getting new validation set. Iteration {}'
                      .format(iter))
                continue

        else:
            break

    return train_inds, test_inds, val_inds


def train_test_dur_split(vds,
                         labelset,
                         train_dur,
                         test_dur,
                         val_dur=None):
    """split a VocalizationDataset

    Parameters
    ----------
    vds : vak.dataset.VocalizationDataset
        a dataset of vocalizations
    labelset : set, list
        of str or int, set of labels for vocalizations.
    train_dur : float
        total duration of training set, in seconds. Default is None.
    val_dur : float
        total duration of validation set, in seconds. Default is None.
    test_dur : float
        total duration of test set, in seconds. Default is None.


    Returns
    -------
    train_vds, test_vds, val_vds
    """
    durs = [voc.duration for voc in vds.voc_list]
    labels = [voc.annotation.labels for voc in vds.voc_list]

    train_inds, test_inds, val_inds = train_test_dur_split_inds(durs=durs,
                                                                labels=labels,
                                                                labelset=labelset,
                                                                train_dur=train_dur,
                                                                test_dur=test_dur,
                                                                val_dur=val_dur)

    train_vds = VocalizationDataset(voc_list=[vds.voc_list[ind] for ind in train_inds])
    test_vds = VocalizationDataset(voc_list=[vds.voc_list[ind] for ind in test_inds])
    if val_inds:
        val_vds = VocalizationDataset(voc_list=[vds.voc_list[ind] for ind in val_inds])
    else:
        val_vds = None

    if val_vds:
        return train_vds, val_vds, test_vds
    else:
        return train_vds, test_vds
