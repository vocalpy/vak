"""tests for vak.split.algorithms.bruteforce module"""
from math import isclose

import numpy as np
import pytest

from vak.split.algorithms import brute_force

# since the algorithm is random, we test multiple times
# yes, this opens up the possibility of non-deterministic failures
NUM_SAMPLES = 10


def is_expected_output(
    train_dur,
    val_dur,
    test_dur,
    labelset,
    durs,
    labels,
    train_inds,
    val_inds,
    test_inds,
):
    """asserts that output from ``bruteforce`` is expected output

    Checks that each split returned by ``bruteforce`` is the
    expected duration, and contains all the labels in ``labelset``.

    Parameters
    ----------
    train_dur
    val_dur
    test_dur
    labelset
    durs
    labels
    train_inds
    val_inds
    test_inds

    Returns
    -------
    expected : bool
        True if output matched expected

    Notes
    -----
    uses specified target durations for splits
    (``train_dur``, ``val_dur``, and ``test_dur``)
    and list of indices that correspond to splits
    found by ``split.algorithms.bruteforce``
    (``train_inds``, ``val_inds``, and ``test_inds``)
    to check that durations of returned splits are:
    - equal to or greater than specified target duration,
      when target duration > 0
    - equal to the total duration of the dataset minus
      the size of the other split,
      when the target duration is set to -1.
      E.g., if the target ``test_dur`` is set to -1,
      the duration of the returned split should be
      approximately the total duration of the dataset
      minus the duration of the returned ``train_inds``.
    When a duration is specified as None, then this function
    asserts that ``bruteforce`` returned None.

    For any split with a target duration (not None),
    this function also checks that the set of labels
    in the split equals the specified ``labelset``.
    """
    for split, dur_in, inds in zip(
        ("train", "val", "test"),
        (train_dur, val_dur, test_dur),
        (train_inds, val_inds, test_inds),
    ):
        if dur_in is not None:
            dur_out = sum([durs[ind] for ind in inds])
            if dur_in > 0:
                assert dur_out >= dur_in
            elif dur_in == -1:
                if split == "train":
                    assert isclose(
                        dur_out, sum(durs) - sum([durs[ind] for ind in test_inds])
                    )
                elif split == "test":
                    assert isclose(
                        dur_out, sum(durs) - sum([durs[ind] for ind in train_inds])
                    )

            all_lbls_this_set = [lbl for ind in inds for lbl in labels[ind]]
            assert labelset == set(all_lbls_this_set)
        else:
            assert inds is None

    assert set(train_inds).isdisjoint(set(test_inds))
    if val_dur is not None:
        assert set(train_inds).isdisjoint(set(val_inds))
        assert set(test_inds).isdisjoint(set(val_inds))

    return True


def test_bruteforce_train_test_val_mock():
    train_dur = 2
    test_dur = 2
    val_dur = 1
    durs = (1, 1, 1, 1, 1)
    labelset = set(list("abcde"))
    labels = [list("abcde") for _ in range(5)]

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = brute_force(
            durs, labels, labelset, train_dur, val_dur, test_dur
        )

        assert is_expected_output(
            train_dur,
            val_dur,
            test_dur,
            labelset,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


def test_bruteforce_train_test_val_cbin(
    audio_cbin_annot_notmat_durs_labels, labelset_notmat
):
    labelset_notmat = set(labelset_notmat)
    durs, labels = audio_cbin_annot_notmat_durs_labels
    train_dur = 35
    val_dur = 20
    test_dur = 35

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = brute_force(
            durs, labels, labelset_notmat, train_dur, val_dur, test_dur
        )

        assert is_expected_output(
            train_dur,
            val_dur,
            test_dur,
            labelset_notmat,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


def test_bruteforce_train_test_val_mat(
    spect_mat_annot_yarden_durs_labels, labelset_yarden
):
    labelset_yarden = set(labelset_yarden)
    durs, labels = spect_mat_annot_yarden_durs_labels
    train_dur = 200
    val_dur = 100
    test_dur = 200

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = brute_force(
            durs, labels, labelset_yarden, train_dur, val_dur, test_dur
        )

        assert is_expected_output(
            train_dur,
            val_dur,
            test_dur,
            labelset_yarden,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


def test_bruteforce_train_neg_one_test_mock():
    train_dur = 2
    val_dur = None
    test_dur = -1
    durs = (1, 1, 1, 1, 1)
    labelset = set(list("abcde"))
    labels = [list("abcde") for _ in range(5)]

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = brute_force(
            durs, labels, labelset, train_dur, val_dur, test_dur
        )

        assert is_expected_output(
            train_dur,
            val_dur,
            test_dur,
            labelset,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


def test_bruteforce_train_neg_one_test_cbin(
    audio_cbin_annot_notmat_durs_labels, labelset_notmat
):
    labelset_notmat = set(labelset_notmat)
    durs, labels = audio_cbin_annot_notmat_durs_labels
    train_dur = 35
    val_dur = None
    test_dur = -1

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = brute_force(
            durs, labels, labelset_notmat, train_dur, val_dur, test_dur
        )

        assert is_expected_output(
            train_dur,
            val_dur,
            test_dur,
            labelset_notmat,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


def test_bruteforce_train_mat(spect_mat_annot_yarden_durs_labels, labelset_yarden):
    labelset_yarden = set(labelset_yarden)
    durs, labels = spect_mat_annot_yarden_durs_labels
    train_dur = 300
    val_dur = None
    test_dur = -1

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = brute_force(
            durs, labels, labelset_yarden, train_dur, val_dur, test_dur
        )

        assert is_expected_output(
            train_dur,
            val_dur,
            test_dur,
            labelset_yarden,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


def test_bruteforce_test_mock():
    train_dur = -1
    test_dur = 2
    val_dur = None
    durs = (1, 1, 1, 1, 1)
    labelset = set(list("abcde"))
    labels = [list("abcde") for _ in range(5)]

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = brute_force(
            durs, labels, labelset, train_dur, val_dur, test_dur
        )

        assert is_expected_output(
            train_dur,
            val_dur,
            test_dur,
            labelset,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


def test_bruteforce_test_cbin(audio_cbin_annot_notmat_durs_labels, labelset_notmat):
    labelset_notmat = set(labelset_notmat)
    durs, labels = audio_cbin_annot_notmat_durs_labels
    train_dur = -1
    val_dur = None
    test_dur = 25

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = brute_force(
            durs, labels, labelset_notmat, train_dur, val_dur, test_dur
        )

        assert is_expected_output(
            train_dur,
            val_dur,
            test_dur,
            labelset_notmat,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


def test_bruteforce_test_mat(spect_mat_annot_yarden_durs_labels, labelset_yarden):
    labelset_yarden = set(labelset_yarden)
    durs, labels = spect_mat_annot_yarden_durs_labels
    train_dur = -1
    val_dur = None
    test_dur = 200

    for _ in range(NUM_SAMPLES):
        train_inds, val_inds, test_inds = brute_force(
            durs, labels, labelset_yarden, train_dur, val_dur, test_dur
        )

        assert is_expected_output(
            train_dur,
            val_dur,
            test_dur,
            labelset_yarden,
            durs,
            labels,
            train_inds,
            val_inds,
            test_inds,
        )


@pytest.mark.parametrize(
    'labels_str',
    [
        # missing labels not in labelset
        'abcd',  # no 'e'
        # extra labels
        'abcdef',
        # none of the labels in labelset
        'ghijkl',
    ]
)
def test_bruteforce_raises(labels_str):
    """Test that ``brute_force`` raises an error
     when ``labels`` does not equal ``labelset``.
     """
    train_dur = 2
    test_dur = 2
    val_dur = 1
    durs = (1, 1, 1, 1, 1)
    labelset = set(list("abcde"))
    labels = [np.array(list(labels_str)) for _ in range(5)]

    with pytest.raises(ValueError):
        brute_force(
            durs, labels, labelset, train_dur, val_dur, test_dur
        )
