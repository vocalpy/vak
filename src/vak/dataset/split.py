import warnings

from .classes import VocalizationDataset
from .utils import _validate_durs
from .splitalgos import brute_force


def train_test_dur_split_inds(durs,
                              labels,
                              labelset,
                              train_dur,
                              test_dur,
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
    train_dur, val_dur, test_dur = _validate_durs(train_dur, val_dur, test_dur, total_dur)

    if -1 not in (train_dur, val_dur, test_dur):
        total_target_dur = sum([dur for dur in (train_dur, test_dur, val_dur) if dur is not None])

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
        train_inds,  val_inds, test_inds = brute_force(durs,
                                                       labels,
                                                       labelset,
                                                       train_dur,
                                                       val_dur,
                                                       test_dur)
    else:
        raise NotImplementedError(
            f'algorithm {algo} not implemented'
        )

    return train_inds, val_inds, test_inds


def train_test_dur_split(vds,
                         labelset,
                         train_dur=None,
                         test_dur=None,
                         val_dur=None):
    """split a VocalizationDataset into training, test, and (optionally) validation sets.

    Parameters
    ----------
    vds : vak.dataset.VocalizationDataset
        a dataset of vocalizations
    labelset : set, list
        of str or int, set of labels for vocalizations.
    train_dur : float
        total duration of training set, in seconds. Default is None
    test_dur : float
        total duration of test set, in seconds. Default is None.
    val_dur : float
        total duration of validation set, in seconds. Default is None.

    Returns
    -------
    train_vds, test_vds, val_vds
    """
    durs = [voc.duration for voc in vds.voc_list]
    vds_dur = sum(durs)
    train_dur, val_dur, test_dur = _validate_durs(train_dur, val_dur, test_dur, vds_dur)

    labels = [voc.annot.labels for voc in vds.voc_list]

    train_inds, val_inds, test_inds = train_test_dur_split_inds(durs=durs,
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
