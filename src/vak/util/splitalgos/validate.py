def validate_durations_convert_nonnegative(train_dur, val_dur, test_dur, vds_dur):
    """helper function to validate durations specified for splits,
    and convert all durations to non-negative numbers,
    so other functions can do the actual splitting.

    First the functions checks for invalid conditions:
        + If train_dur, val_dur, and test_dur are all None, a ValueError is raised.
        + If any of train_dur, val_dur, or test_dur have a negative value that is not -1, an
          ValueError is raised. -1 is interpreted differently as explained below.
        + If only val_dur is specified, this raises a ValueError; not clear what durations of training
          and test set should be.

    Then, if either train_dur or test_dur are None, they are set to 0. None means user did not specify a value.

    Finally the function validates that the sum of the specified split durations is not greater than
    the the total duration of the dataset, `vds_dur`.

    If any split is specified as -1, this value is interpreted as "first get the
    split for the set with a value specified, then use the remainder of the dataset in the split
    whose duration is set to -1". The duration for the split specified as -1 will be computed by
    first summing the value of the other two splits, validating that they are not larger than
    the total duration of the dataset, and finally subtracting off the combined duration of the
    other two.

    Parameters
    ----------
    train_dur : int, float
        Target duration for training set, in seconds.
    val_dur : int, float
        Target duration for validation set, in seconds.
    test_dur : int, float
        Target duration for test set, in seconds.
    vds_dur : int, float
        Total duration of Dataset.

    Returns
    -------
    train_dur, val_dur, test_dur : int, float
    """
    if val_dur and (train_dur is None and test_dur is None):
        raise ValueError(
            'cannot specify only val_dur, unclear how to split dataset into training and test sets'
        )

    # make a dict so we can refer to variable by name in loop
    split_durs = {
        'train': train_dur,
        'val': val_dur,
        'test': test_dur,
    }
    if all([dur is None for dur in split_durs.values()]):
        raise ValueError("train_dur, val_dur, and test_dur were all None; must specify at least train_dur or test_dur")

    if not all([dur >= 0 or dur == -1 for dur in split_durs.values() if dur is not None]):
        raise ValueError(
            "all durations for split must be real non-negative number or "
            "set to -1 (meaning 'use the remaining dataset)"
        )

    if sum([split_dur == -1 for split_dur in split_durs.values()]) > 1:
        raise ValueError(
            'cannot specify duration of more than one split as -1, unclear how to calculate durations of splits.'
        )

    # set any None values for durations to 0; no part of dataset will go to that split
    for split_name in split_durs.keys():
        if split_durs[split_name] is None:
            split_durs[split_name] = 0

    if -1 in split_durs.values():
        total_other_splits_dur = sum([dur for dur in split_durs.values() if dur is not -1])

        if total_other_splits_dur > vds_dur:
            raise ValueError(
                'One dataset split duration was specified as -1, but the total of the other durations specified, '
                f'{total_other_splits_dur} s, is greater than total duration of Dataset, {vds_dur}.'
            )
        else:
            remainder = vds_dur - total_other_splits_dur
            for split_name in split_durs.keys():
                if split_durs[split_name] == -1:
                    split_durs[split_name] = remainder

    # validate one last time now that we have real non-negative values for all split durations
    total_splits_dur = sum(split_durs.values())

    if total_splits_dur > vds_dur:
        raise ValueError(
            f'Total of the split durations specified, {total_splits_dur} s, '
            f'is greater than total duration of dataset, {vds_dur}.'
        )

    return split_durs['train'], split_durs['val'], split_durs['test']
