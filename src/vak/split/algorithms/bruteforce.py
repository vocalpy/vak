import random
import itertools
import logging

from .validate import validate_durations_convert_nonnegative


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

    The durations of the datasets created using the returned indices will be
    *greater than* or equal to the durations specified.

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
    Starts by ensuring that each label is represented in each set and then adds files to reach the required
    durations.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    sum_durs = sum(durs)
    train_dur, val_dur, test_dur = validate_durations_convert_nonnegative(train_dur, val_dur, test_dur, sum_durs)
    total_target_dur = sum([dur for dur in (train_dur, val_dur, test_dur) if dur is not None])

    durs_labels_list = list(zip(durs, labels))
    iter = 1
    all_labels_err = ('Did not successfully divide data into training, '
                      'validation, and test sets of sufficient duration '
                      f'after {max_iter} iterations.'
                      ' Try increasing the total size of the data set.')

    # ---- outer loop that repeats until we successfully split our reach max number of iters ---------------------------
    while 1:
        train_inds = []
        test_inds = []
        if val_dur > 0:
            val_inds = []
        else:
            val_inds = None

        total_train_dur = 0
        total_val_dur = 0
        total_test_dur = 0

        durs_labels_inds = list(range(len(durs_labels_list)))

        choice = []
        if train_dur > 0:
            choice.append('train')
            lset_train = set()
        else:
            lset_train = None
        if test_dur > 0:
            choice.append('test')
            lset_test = set()
        else:
            lset_test = None
        if val_dur:
            choice.append('val')
            lset_val = set()
        else:
            lset_val = None

        # ---- make sure each split has at least one instance of each label
        for label in sorted(labelset):
            label_inds = [ind for ind, labels in enumerate(labels)
                          if label in labels
                          and ind in durs_labels_inds]
            if len(label_inds) < len(choice):
                raise ValueError(
                    f'unable to split dataset so that each split has an instance of label: {label}.'
                    f'There were only {len(label_inds)} files with that label, '
                    f'but there are {len(choice)} splits.'
                )

            random.shuffle(label_inds)
            if train_dur > 0 and label not in lset_train:
                try:
                    ind = label_inds.pop()
                    train_inds.append(ind)
                    total_train_dur += durs[ind]
                    lset_train = lset_train.union(set(labels[ind]))
                    durs_labels_inds.remove(ind)
                except IndexError:
                    if len(label_inds) == 0:
                        logger.debug(
                            'Ran out of elements while dividing dataset into subsets of specified durations.'
                            f'Iteration {iter}'
                        )
                        iter += 1
                        break  # do next iteration
                    else:
                        # something else happened, re-raise error
                        raise

            if test_dur > 0 and label not in lset_test:
                try:
                    ind = label_inds.pop()
                    test_inds.append(ind)
                    total_test_dur += durs[ind]
                    lset_test = lset_test.union(set(labels[ind]))
                    durs_labels_inds.remove(ind)
                except IndexError:
                    if len(label_inds) == 0:
                        logger.debug(
                            'Ran out of elements while dividing dataset into subsets of specified durations.'
                            f'Iteration {iter}'
                        )
                        iter += 1
                        break  # do next iteration
                    else:
                        # something else happened, re-raise error
                        raise

            if val_dur > 0 and label not in lset_val:
                try:
                    ind = label_inds.pop()
                    val_inds.append(ind)
                    total_val_dur += durs[ind]
                    lset_val = lset_val.union(set(labels[ind]))
                    durs_labels_inds.remove(ind)
                except IndexError:
                    if len(label_inds) == 0:
                        logger.debug(
                            'Ran out of elements while dividing dataset into subsets of specified durations.'
                            f'Iteration {iter}'
                        )
                        iter += 1
                        break  # do next iteration
                    else:
                        # something else happened, re-raise error
                        raise

        random.shuffle(durs_labels_inds)
        if train_dur > 0 and total_train_dur >= train_dur:
            choice.remove('train')
        if test_dur > 0 and total_test_dur >= test_dur:
            choice.remove('test')
        if val_dur > 0 and total_val_dur >= val_dur:
            choice.remove('val')

        if len(choice) == 0:
            finished = True
        else:
            finished = False

        # ---- inner loop that actually does split -----------------
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
            if train_dur > 0:
                train_tups = [durs_labels_list[ind] for ind in train_inds]  # tup = a tuple
                train_labels = itertools.chain.from_iterable(
                    [tup[1] for tup in train_tups])
                train_labelset = set(train_labels)  # make set to get unique values
                if train_labelset != set(labelset):
                    iter += 1
                    if iter > max_iter:
                        raise ValueError(all_labels_err)
                    else:
                        logger.debug(
                            'Train labels did not contain all labels in labelset. '
                            f'Getting new training set. Iteration {iter}'
                        )
                        continue

            if test_dur > 0:
                test_tups = [durs_labels_list[ind] for ind in test_inds]
                test_labels = itertools.chain.from_iterable(
                    [tup[1] for tup in test_tups])
                test_labelset = set(test_labels)
                if test_labelset != set(labelset):
                    iter += 1
                    if iter > max_iter:
                        raise ValueError(all_labels_err)
                    else:
                        logger.debug(
                            'Test labels did not contain all labels in labelset. '
                            f'Getting new test set. Iteration {iter}'
                        )
                        continue

            if val_dur > 0:
                val_tups = [durs_labels_list[ind] for ind in val_inds]
                val_labels = itertools.chain.from_iterable(
                    [tup[1] for tup in val_tups])
                val_labelset = set(val_labels)
                if val_labelset != set(labelset):
                    iter += 1
                    if iter > max_iter:
                        raise ValueError(all_labels_err)
                    else:
                        logger.debug(
                            'Validation labels did not contain all labels in labelset. '
                            f'Getting new validation set. Iteration {iter}'
                        )
                        continue

            # successfully split
            break

        elif finished is False:
            continue

    return train_inds, val_inds, test_inds
