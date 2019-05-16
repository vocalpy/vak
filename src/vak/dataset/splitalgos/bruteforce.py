import random
import itertools
import logging


def brute_force(durs, labels, labelset, train_dur, test_dur, val_dur):
    """randomly partition vocalizations into training, test, and validation
    sets.

    A 'brute force' algorithm that just randomly assigns vocalizations to
    a set and iterates until it finds some partition where each set has
    instances of all classes of label.

    Usually works but is not necessarily the most efficient."""
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

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
                    logger.debug(
                        'Ran out of elements while dividing dataset into subsets of specified durations.'
                        f'Iteration {iter}'
                    )
                    iter += 1
                    break  # do next iteration
                else:
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
                logger.debug(
                    'Train labels did not contain all labels in labelset. '
                    f'Getting new training set. Iteration {iter}'
                )
                continue
        elif test_labels != set(labelset):
            iter += 1
            if iter > 1000:
                raise ValueError(all_labels_err)
            else:
                logger.debug(
                    'Test labels did not contain all labels in labelset. '
                    f'Getting new test set. Iteration {iter}'
                )
                continue
        elif val_dur is not None and val_labels != set(labelset):
            iter += 1
            if iter > 1000:
                raise ValueError(all_labels_err)
            else:
                logger.debug(
                    'Validation labels did not contain all labels in labelset. '
                    f'Getting new validation set. Iteration {iter}'
                )
                continue

        else:
            break

    return train_inds, test_inds, val_inds
