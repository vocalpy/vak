import logging
import random

from .validate import validate_split_durations


def brute_force(durs, labels, labelset, train_dur, val_dur, test_dur, max_iter=5000):
    """finds indices that split (labels, durations) tuples into training,
    test, and validation sets of specified durations, with the set of unique labels
    in each dataset equal to the specified labelset.

    The durations of the datasets created using the returned indices will be
    *greater than* or equal to the durations specified.

    Must specify a positive value for one of {train_dur, test_dur}.
    The other value can be specified as '-1' which is interpreted as
    "use the remainder of the dataset for this split,
    after finding indices for the set with a specified duration".

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
    logger.setLevel("INFO")

    sum_durs = sum(durs)
    train_dur, val_dur, test_dur = validate_split_durations(
        train_dur, val_dur, test_dur, sum_durs
    )
    target_split_durs = dict(
        zip(("train", "val", "test"), (train_dur, val_dur, test_dur))
    )

    if not len(durs) == len(labels):
        raise ValueError(
            "length of list of durations did not equal length of list of labels; "
            "should be same length since "
            "each duration of a vocalization corresponds to the labels from its annotations.\n"
            f"Length of durations: {len(durs)}. Length of labels: {len(labels)}"
        )

    iter = 1
    all_labels_err = (
        "Did not successfully divide data into training, "
        "validation, and test sets of sufficient duration "
        f"after {max_iter} iterations. "
        "Try increasing the total size of the data set."
    )

    # ---- outer loop that repeats until we successfully split our reach max number of iters ---------------------------
    while 1:
        # list of indices we use to index into both `durs` and `labels`
        durs_labels_inds = list(
            range(len(labels))
        )  # we checked len(labels) == len(durs) above

        # when making `split_inds`, "initialize" the dict with all split names, by using target_split_durs
        # so we don't get an error when indexing into dict in return statement below
        split_inds = {split_name: [] for split_name in target_split_durs.keys()}
        total_split_durs = {split_name: 0 for split_name in target_split_durs.keys()}
        split_labelsets = {split_name: set() for split_name in target_split_durs.keys()}

        # list of split 'choices' we use when randomly adding indices to splits
        choice = []
        for split_name in target_split_durs.keys():
            if target_split_durs[split_name] > 0 or target_split_durs[split_name] == -1:
                choice.append(split_name)

        # ---- make sure each split has at least one instance of each label --------------------------------------------
        for label_from_labelset in sorted(labelset):
            label_inds = [
                ind for ind in durs_labels_inds if label_from_labelset in labels[ind]
            ]

            random.shuffle(label_inds)
            for split_name in target_split_durs.keys():
                if (
                    target_split_durs[split_name] > 0
                    or target_split_durs[split_name] == -1
                ) and label_from_labelset not in split_labelsets[split_name]:
                    try:
                        ind = label_inds.pop()
                        split_inds[split_name].append(ind)
                        total_split_durs[split_name] += durs[ind]
                        split_labelsets[split_name] = split_labelsets[split_name].union(
                            set(labels[ind])
                        )
                        durs_labels_inds.remove(ind)
                    except IndexError:
                        if len(label_inds) == 0:
                            logger.debug(
                                "Ran out of elements while dividing dataset into subsets of specified durations."
                                f"Iteration {iter}"
                            )
                            iter += 1
                            break  # do next iteration
                        else:
                            # something else happened, re-raise error
                            raise

        for split_name in target_split_durs.keys():
            if (
                target_split_durs[split_name] > 0
                and total_split_durs[split_name] >= target_split_durs[split_name]
            ):
                choice.remove(split_name)

        if len(choice) == 0:
            finished = True
        else:
            finished = False

        # ---- inner loop that actually does split ---------------------------------------------------------------------
        random.shuffle(durs_labels_inds)
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
                        "Ran out of elements while dividing dataset into subsets of specified durations."
                        f"Iteration {iter}"
                    )
                    iter += 1
                    break  # do next iteration
                else:
                    # something else happened, re-raise error
                    raise

            which_set = random.randint(0, len(choice) - 1)
            split_name = choice[which_set]
            split_inds[split_name].append(ind)
            total_split_durs[split_name] += durs[ind]
            if (
                target_split_durs[split_name] > 0
                and total_split_durs[split_name] >= target_split_durs[split_name]
            ):
                choice.remove(split_name)
            elif target_split_durs[split_name] == -1:
                # if this split is -1 and other split is already "finished"
                if (split_name == "test" and "train" not in choice) or (
                    split_name == "train" and "test" not in choice
                ):
                    # just add all remaining inds to this split
                    split_inds[split_name].extend(durs_labels_inds)
                    choice.remove(split_name)

            if len(choice) < 1:  # list is empty, we popped off all the choices
                for split_name in target_split_durs.keys():
                    if target_split_durs[split_name] > 0:
                        if total_split_durs[split_name] < target_split_durs[split_name]:
                            raise ValueError(
                                "Loop to find splits completed, "
                                f"but total duration of '{split_name}' split, "
                                f"{total_split_durs[split_name]} seconds, "
                                f"is less than target duration specified: {target_split_durs[split_name]} seconds."
                            )
                else:
                    finished = True
                    break

        if iter > max_iter:
            raise ValueError(
                "Could not find subsets of sufficient duration in "
                f"less than {max_iter} iterations."
            )

        # make sure that each split contains all unique labels in labelset
        if finished is True:
            for split_name in target_split_durs.keys():
                if (
                    target_split_durs[split_name] > 0
                    or target_split_durs[split_name] == -1
                ):
                    split_labels = [
                        label for ind in split_inds[split_name] for label in labels[ind]
                    ]
                    split_labelset = set(split_labels)
                    if split_labelset != set(labelset):
                        iter += 1
                        if iter > max_iter:
                            raise ValueError(all_labels_err)
                        else:
                            logger.debug(
                                f"Set of unique labels in '{split_name}' split did not equal specified labelset. "
                                f"Getting new '{split_name}' split. Iteration: {iter}"
                            )
                            continue

            # successfully split
            break

        elif finished is False:
            continue

    split_inds = {
        split_name: (inds if inds else None) for split_name, inds in split_inds.items()
    }

    return split_inds["train"], split_inds["val"], split_inds["test"]
