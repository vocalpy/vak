import random
import itertools
import logging

from ..utils import _validate_durs


def brute_force(durs,
                labels,
                labelset,
                train_dur,
                val_dur,
                test_dur,
                max_iter=5000):
    """finds indices that split (labels, durations) tuples into training,
    test, and validation sets of specified durations, with the set of unique labels
    in each dataset equal to the specified labelset.

    The durations of the datasets created using the returned indices will be *greater than* or equal to
    the durations specified.

    Must specify a positive value for one of {train_dur, test_dur}.
    The other value can be specified as '-1' which is interpreted as "use the
    remainder of the dataset for this split after finding indices for the set with a specified duration".
    If only one of {train_dur, test_dur} is specified, the other defaults to '-1'.

    Parameters
    ----------
    durs : list
        of durations of vocalizations
    labels : list
        of labels from vocalizations
    labelset : set
        of labels
    train_dur : int, float
        Target duration for training set, in seconds.
    val_dur : int, float
        Target duration for validation set, in seconds.
    test_dur : int, float
        Target duration for test set, in seconds.
    max_iter : int
        maximum number of iterations to attempt to find indices. Default is 5000.

    Returns
    -------
    train_inds, val_inds, test_inds : list
        of int, the indices that will split datasets

    Notes
    -----
    A 'brute force' algorithm that just randomly assigns indices to a set,
    and iterates until it finds some partition where each set has instances of all classes of label.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    sum_durs = sum(durs)
    train_dur, val_dur, test_dur = _validate_durs(train_dur, val_dur, test_dur, sum_durs)

    if -1 not in (train_dur, test_dur):
        total_target_dur = sum([dur for dur in (train_dur, val_dur, test_dur) if dur is not None])
    else:
        total_target_dur = sum_durs

    durs_labels_list = list(zip(durs, labels))
    iter = 1
    all_labels_err = ('Did not successfully divide data into training, '
                      'validation, and test sets of sufficient duration '
                      f'after {max_iter} iterations.'
                      ' Try increasing the total size of the data set.')

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

        durs_labels_inds = list(range(len(durs_labels_list)))
        random.shuffle(durs_labels_inds)

        finished = False

        if -1 in (train_dur, test_dur):
            if train_dur == -1:
                choice = ['test', 'train']
            elif test_dur == -1:
                choice = ['train', 'test']

            while finished is False:
                try:
                    ind = durs_labels_inds.pop()
                except IndexError:
                    if len(durs_labels_inds) == 0:
                        logger.debug(
                            'Ran out of elements while dividing dataset into subsets of specified durations.'
                            f'Iteration {iter}'
                        )
                        iter += 1
                        break  # do next iteration
                    else:
                        # something else happened, re-raise error
                        raise
                which_set = choice[0]
                if which_set == 'train':
                    train_inds.append(ind)
                    total_train_dur += durs_labels_list[ind][0]  # ind 0 is duration
                    if train_dur != -1 and total_train_dur >= train_dur:
                        choice.pop(choice.index('train'))
                elif which_set == 'test':
                    test_inds.append(ind)
                    total_test_dur += durs_labels_list[ind][0]  # ind 0 is duration
                    if test_dur != -1 and total_test_dur >= test_dur:
                        choice.pop(choice.index('test'))

                if len(durs_labels_inds) == 0:
                    finished = True
                    break

        else:
            choice = ['train', 'test']
            if val_dur:
                choice.append('val')

            while finished is False:
                # pop durations off list and append to randomly-chosen
                # list, either train, val, or test set.
                # Do this until the total duration for each data set is equal
                # to or greater than the target duration for each set.
                try:
                    ind = durs_labels_inds.pop()
                except IndexError:
                    if len(durs_labels_inds) == 0:
                        logger.debug(
                            'Ran out of elements while dividing dataset into subsets of specified durations.'
                            f'Iteration {iter}'
                        )
                        iter += 1
                        break  # do next iteration
                    else:
                        # something else happened, re-raise error
                        raise

                which_set = random.randint(0, len(choice) - 1)
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
                    total_all_durs = sum([dur for dur in (total_train_dur, total_val_dur, total_test_dur)
                                          if dur is not None])
                    if total_all_durs < total_target_dur:
                        raise ValueError(
                            f'Loop to find subsets completed but total duration of subsets, {total_all_durs} seconds, '
                            f'is less than total duration specified: {total_target_dur} seconds.')
                    else:
                        finished = True
                        break

            if iter > max_iter:
                raise ValueError('Could not find subsets of sufficient duration in '
                                 f'less than {max_iter} iterations.')

        if finished is True:
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
                if iter > max_iter:
                    raise ValueError(all_labels_err)
                else:
                    logger.debug(
                        'Train labels did not contain all labels in labelset. '
                        f'Getting new training set. Iteration {iter}'
                    )
                    continue
            elif test_labels != set(labelset):
                iter += 1
                if iter > max_iter:
                    raise ValueError(all_labels_err)
                else:
                    logger.debug(
                        'Test labels did not contain all labels in labelset. '
                        f'Getting new test set. Iteration {iter}'
                    )
                    continue
            elif val_dur is not None and val_labels != set(labelset):
                iter += 1
                if iter > max_iter:
                    raise ValueError(all_labels_err)
                else:
                    logger.debug(
                        'Validation labels did not contain all labels in labelset. '
                        f'Getting new validation set. Iteration {iter}'
                    )
                    continue
            else:
                break
        elif finished is False:
            continue

    return train_inds, val_inds, test_inds
