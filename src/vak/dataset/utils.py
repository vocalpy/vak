class OnlyValDurError(Exception):
    pass


class InvalidDurationError(Exception):
    pass


class SplitsDurationGreaterThanDatasetDurationError(Exception):
    pass


def _validate_durs(train_dur, val_dur, test_dur, vds_dur):
    """helper function to validate durations specified for splits

    If train_dur, val_dur, and test_dur are all None, a ValueError is raised.

    If any of train_dur, val_dur, or test_dur have a negative value that is not -1, an
    InvalidDurationError is raised. -1 is interpreted differently as explained below.

    If all three have non-negative values, this function simply checks that their sum is not
    greater than vds_dur. If this is True, it returns them unchanged. If the total sum *is*
    greater than vds_dur, an error is raised (SplitsDurationGreaterThanDatasetDurationError).

    If only val_dur is specified, this raises a ValDurError; not clear what durations of training
    and test set should be.

    If only train_dur is specified, then test_dur is set to -1; similarly if oly test_dur is
    specified, then train_dur is set to -1. Other functions interpret this as "first get the
    split for the set with a value specified, then use the remainder of the dataset in the split
    whose duration is set to -1".

    Parameters
    ----------
    train_dur : int, float
        Target duration for training set, in seconds.
    val_dur : int, float
        Target duration for validation set, in seconds.
    test_dur : int, float
        Target duration for test set, in seconds.
    vds_dur : int, float
        Total duration of VocalizationDataset.

    Returns
    -------
    train_dur, val_dur, test_dur : int, float
    """
    if all([dur is None for dur in (train_dur, val_dur, test_dur)]):
        raise ValueError("train_dur, val_dur, and test_dur were all None; must specify at least train_dur or test_dur")

    else:
        if not all([dur > 0 or dur == -1 for dur in (train_dur, val_dur, test_dur) if dur is not None]):
            raise InvalidDurationError("all durations for split must be real positive number or "
                                       "set to -1 (meaning 'use the remaining dataset)")

        if val_dur and train_dur is None and test_dur is None:
            raise OnlyValDurError(
                'cannot specify only val_dur, unclear how to split dataset into training and test sets'
            )

        if train_dur:
            if (test_dur is None and val_dur is None) or (val_dur and test_dur is None):
                test_dur = -1  # keep val_dur None

        elif test_dur:  # and train_dur was None
            if (train_dur is None and val_dur is None) or (val_dur and train_dur is None):
                train_dur = -1  # keep val_dur None

        if -1 not in (train_dur, val_dur, test_dur):
            total_splits_dur = sum([dur for dur in (train_dur, val_dur, test_dur) if dur is not None])
            if total_splits_dur > vds_dur:
                raise SplitsDurationGreaterThanDatasetDurationError(
                    f'total of durations specified for dataset split, {total_splits_dur} s, '
                    f'is greater than total duration of VocalizationDataset, {vds_dur}.'
                )

    return train_dur, val_dur, test_dur
