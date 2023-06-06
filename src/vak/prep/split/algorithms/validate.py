def validate_split_durations(train_dur, val_dur, test_dur, dataset_dur):
    """helper function to validate durations specified for splits,
    so other functions can do the actual splitting.

    First the functions checks for invalid conditions:
        + If train_dur, val_dur, and test_dur are all None, a ValueError is raised.
        + If any of train_dur, val_dur, or test_dur have a negative value that is not -1, an
          ValueError is raised. -1 is interpreted differently as explained below.
        + If only val_dur is specified, this raises a ValueError; not clear what durations of training
          and test set should be.

    Then, if either train_dur or test_dur are None, they are set to 0. None means user did not specify a value.

    Finally the function validates that the sum of the specified split durations is not greater than
    the the total duration of the dataset, `dataset_dur`.

    If any split is specified as -1, this value is interpreted as "first get the
    split for the set with a value specified, then use the remainder of the dataset in the split
    whose duration is set to -1". Functions that do the splitting have to "know"
    about this meaning of -1, so this validation function does not modify the value.

    Parameters
    ----------
    train_dur : int, float
        Target duration for training set split, in seconds.
    val_dur : int, float
        Target duration for validation set, in seconds.
    test_dur : int, float
        Target duration for test set, in seconds.
    dataset_dur : int, float
        Total duration of dataset of vocalizations that will be split.

    Returns
    -------
    train_dur, val_dur, test_dur : int, float
    """
    if val_dur and (train_dur is None and test_dur is None):
        raise ValueError(
            "cannot specify only val_dur, unclear how to split dataset into training and test sets"
        )

    # make a dict so we can refer to variable by name in loop
    split_durs = {
        "train": train_dur,
        "val": val_dur,
        "test": test_dur,
    }
    if all([dur is None for dur in split_durs.values()]):
        raise ValueError(
            "train_dur, val_dur, and test_dur were all None; must specify at least train_dur or test_dur"
        )

    if not all(
        [dur >= 0 or dur == -1 for dur in split_durs.values() if dur is not None]
    ):
        raise ValueError(
            "all durations for split must be real non-negative number or "
            "set to -1 (meaning 'use the remaining dataset)"
        )

    if sum([split_dur == -1 for split_dur in split_durs.values()]) > 1:
        raise ValueError(
            "cannot specify duration of more than one split as -1, unclear how to calculate durations of splits."
        )

    # set any None values for durations to 0; no part of dataset will go to that split
    for split_name in split_durs.keys():
        if split_durs[split_name] is None:
            split_durs[split_name] = 0

    if -1 in split_durs.values():
        total_other_splits_dur = sum([dur for dur in split_durs.values() if dur != -1])

        if total_other_splits_dur > dataset_dur:
            raise ValueError(
                "One dataset split duration was specified as -1, but the total of the other durations specified, "
                f"{total_other_splits_dur} s, is greater than total duration of Dataset, {dataset_dur}."
            )
    else:  # if none of the target durations are -1
        total_splits_dur = sum(split_durs.values())

        if total_splits_dur > dataset_dur:
            raise ValueError(
                f"Total of the split durations specified, {total_splits_dur} s, "
                f"is greater than total duration of dataset, {dataset_dur}."
            )

    return split_durs["train"], split_durs["val"], split_durs["test"]
