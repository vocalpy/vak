import warnings

from crowsetta import Transcriber
import numpy as np

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
        on the algorithms, see the docstrings, e.g., vak.io.splitalgos.brute_force
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


def train_test_dur_split(vak_df,
                         labelset,
                         train_dur=None,
                         test_dur=None,
                         val_dur=None):
    """split a dataset of vocalizations into training, test, and (optionally) validation subsets,
    specified by their duration.

    Takes dataset represented as a pandas DataFrame and adds a 'split' column that assigns each
    row to 'train', 'val', 'test', or 'None'.

    Parameters
    ----------
    vak_df : pandas.Dataframe
        a dataset of vocalizations.
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
    vak_df : pandas.Dataframe
        with 'split' column added, that assigns each vocalization (row) to a subset, i.e., train, validation, or test.
        If the vocalization was not added to one of the subsets, its value for 'split' will be 'None'.

    Notes
    -----
    uses the function `vak.dataset.split.train_test_dur_split_inds` to find indices for each subset.
    """
    annot_format = vak_df['annot_format'].unique()
    if len(annot_format) == 1:
        annot_format = annot_format.item()
        # if annot_format is None, throw an error -- otherwise continue on and try to use it
        if annot_format is None:
            raise ValueError(
                'unable to load labels for dataset, the annot_format is None'
            )
    elif len(annot_format) > 1:
        raise ValueError(
            f'unable to load labels for dataset, found multiple annotation formats: {annot_format}'
        )
    # TODO: change this keyword argument to annot_format when changing dependency to crowsetta 2.0
    scribe = Transcriber(annot_format=annot_format)
    labels = [scribe.from_file(annot_file=annot_file).seq.labels
              for annot_file in vak_df['annot_path'].values]

    durs = vak_df['duration'].values
    total_dataset_dur = durs.sum()
    train_dur, val_dur, test_dur = _validate_durs(train_dur, val_dur, test_dur, total_dataset_dur)

    train_inds, val_inds, test_inds = train_test_dur_split_inds(durs=durs,
                                                                labels=labels,
                                                                labelset=labelset,
                                                                train_dur=train_dur,
                                                                test_dur=test_dur,
                                                                val_dur=val_dur)

    # start off with all elements set to 'None'
    # so we don't have to change any that are not assigned to one of the subsets to 'None' after
    split_col = np.asarray(['None' for _ in range(len(vak_df))], dtype='object')
    split_zip = zip(
        ['train', 'val', 'test'],
        [train_inds, val_inds, test_inds]
    )
    for split_name, split_inds in split_zip:
        if split_inds is not None:
            split_col[split_inds] = split_name

    # add split column to dataframe
    vak_df['split'] = split_col

    return vak_df
