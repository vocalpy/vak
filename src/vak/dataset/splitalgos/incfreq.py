import random
import itertools
import logging


def inc_freq(durs, labels, labelset, train_dur, test_dur, val_dur):
    """randomly partition vocalizations into training, test, and validation
    sets

    This algorithm finds the unique set of labels for the vocalizations, then
    sorts them in order of increasing frequency. It then partitions the vocalizations
    into sets starting with the label that occurs *least* frequently. This is a
    'fail-early' approach -- if there's not enough instances of some vocalization,
    we will see that explicitly and be able to error out with an informative message"""
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    # map each label in labelset to list of indices where it occurs in labels
    # (the list of labels from vocalizations)
    labelset_where = {}
    for lbl in labelset:
        labelset_where[lbl] = []
    for lbls_ind, lbls_list in enumerate(labels):
        lbls_set = set(lbls_list)
        for lbl in lbls_set:
            labelset_where[lbl].append(lbls_ind)

    # map this to (duration, labels) tuples for when we do random partitioning in main loop
    labelset_durs_labels = {}
    for lbl, inds in labelset_where.items():
        lbl_durs = [durs[ind] for ind in inds]
        lbl_lbls = [labels[ind] for ind in inds]
        labelset_durs_labels[lbl] = (lbl_durs, lbl_lbls)

    # now count number of occurrences of each label
    labelset_num_occurs = {}
    for lbl, inds in labelset_where.items():
        labelset_num_occurs[lbl] = len(inds)

    labelset_num_occurs = sorted(
        # use value as key for sorting, convert to float so its numeric not alphabetic sort
        labelset_num_occurs.items(), key=lambda kv: float(kv[1])
    )
    # we use this to loop through labelset in order of increasing occurrence below
    labelset_incfreq = [tup[0] for tup in labelset_num_occurs]

    if val_dur is None:
        total_target_dur = sum([train_dur,
                                test_dur])
    elif val_dur is not None:
        total_target_dur = sum([train_dur,
                                test_dur,
                                val_dur])

    iter = 1
    all_labels_err = ('Did not successfully divide data into training, '
                      'validation, and test sets of sufficient duration '
                      'after 1000 iterations.'
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

        choice = ['train', 'test']
        if val_dur:
            choice.append('val')

        least_freq = labelset_num_occurs[0][0]
        least_freq_num = labelset_num_occurs[0][1]
        if least_freq_num < len(choice):
            raise ValueError(
                f'number of occurrences of least frequent label class, {least_freq}, is {least_freq_num}, '
                f'which is less than the number of datasets: {len(choice)}.\nCan not split datasets and have '
                f'each split contain this class.'
            )

        while 1:
            for label in labelset_incfreq:
                durs_labels_tup = labelset_durs_labels[label]
                durs_labels_inds = list(range(len(durs_labels_tup[0])))
                random.shuffle(durs_labels_inds)

                # pop durations off list and append to randomly-chosen
                # list, either train, val, or test set.
                # Do this until the total duration for each data set is equal
                # to or greater than the target duration for each set.
                try:
                    ind = durs_labels_inds.pop()
                except IndexError:
                    if len(durs_labels_inds) == 0:
                        logger.debug(
                            'Ran out of elements while dividing dataset into subsets of '
                            f'specified durations. Iteration {iter}'
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
