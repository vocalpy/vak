import warnings

from .classes import VocalizationDataset
from .splitalgos import brute_force


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
        if not all([dur >= 0 or dur == -1 for dur in (train_dur, val_dur, test_dur) if dur is not None]):
            raise InvalidDurationError("all durations for split must be non-negative integers or "
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


def train_test_dur_split_inds(durs,
                              labels,
                              labelset,
                              train_dur=None,
                              test_dur=None,
                              val_dur=None,
                              algo='brute_force'):
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
    algo : str
        algorithm to use. One of {'brute_force', 'inc_freq'}. Default is 'brute_force'. For more information
        on the algorithms, see the docstrings, e.g., vak.dataset.splitalgos.brute_force
.
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

    if algo == 'brute_force':
        train_inds, test_inds, val_inds = brute_force(durs,
                                                      labels,
                                                      labelset,
                                                      train_dur,
                                                      test_dur,
                                                      val_dur)
    else:
        raise NotImplementedError(
            f'algorithm {algo} not implemented'
        )

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
    labels = [voc.annot.labels for voc in vds.voc_list]

    train_inds, test_inds, val_inds = train_test_dur_split_inds(durs=durs,
                                                                labels=labels,
                                                                labelset=labelset,
                                                                train_dur=train_dur,
                                                                test_dur=test_dur,
                                                                val_dur=val_dur)

    train_vds = VocalizationDataset(voc_list=[vds.voc_list[ind] for ind in train_inds], labelset=labelset)
    test_vds = VocalizationDataset(voc_list=[vds.voc_list[ind] for ind in test_inds], labelset=labelset)
    if val_inds:
        val_vds = VocalizationDataset(voc_list=[vds.voc_list[ind] for ind in val_inds], labelset=labelset)
    else:
        val_vds = None

    if val_vds:
        return train_vds, val_vds, test_vds
    else:
        return train_vds, test_vds
