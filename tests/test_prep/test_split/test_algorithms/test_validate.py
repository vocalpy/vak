import pytest

from vak.split.algorithms.validate import validate_split_durations


@pytest.mark.parametrize(
    "train_dur_in, val_dur_in, test_dur_in, dataset_dur, train_dur_expected, val_dur_expected, test_dur_expected",
    [
        (100, 25, 75, 200, 100, 25, 75),
        (100, None, -1, 200, 100, 0, -1),
        (-1, None, 100, 200, -1, 0, 100),
        (100, 20, -1, 200, 100, 20, -1),
        (-1, 20, 100, 200, -1, 20, 100),
        (100, None, None, 200, 100, 0, 0),
        (None, None, 100, 200, 0, 0, 100),
    ],
)
def test_validate_durs(
    train_dur_in,
    val_dur_in,
    test_dur_in,
    dataset_dur,
    train_dur_expected,
    val_dur_expected,
    test_dur_expected,
):
    train_dur_out, val_dur_out, test_dur_out = validate_split_durations(
        train_dur_in, val_dur_in, test_dur_in, dataset_dur
    )
    assert all(
        [
            train_dur_out == train_dur_expected,
            val_dur_out == val_dur_expected,
            test_dur_out == test_dur_expected,
        ]
    )


def test_validate_durs_all_durs_none_raises():
    train_dur_in = None
    val_dur_in = None
    test_dur_in = None
    dataset_dur = 200
    with pytest.raises(ValueError):
        # because we have to specify at least one of train_dur or test_dur
        validate_split_durations(train_dur_in, val_dur_in, test_dur_in, dataset_dur)


def test_validate_durs_val_only_raises():
    train_dur_in = None
    val_dur_in = 100
    test_dur_in = None
    dataset_dur = 200
    # because we only specified duration for validation set
    with pytest.raises(ValueError):
        validate_split_durations(train_dur_in, val_dur_in, test_dur_in, dataset_dur)


def test_validate_durs_negative_dur_raises():
    train_dur_in = -2
    test_dur_in = None
    val_dur_in = 100
    dataset_dur = 200
    # because negative duration is invalid
    with pytest.raises(ValueError):
        validate_split_durations(train_dur_in, val_dur_in, test_dur_in, dataset_dur)


def test_validate_durs_total_splits_greater_than_dataset_duration_raises():
    train_dur_in = 100
    test_dur_in = 100
    val_dur_in = 100
    dataset_dur = 200
    # because total splits duration is greater than dataset duration
    with pytest.raises(ValueError):
        validate_split_durations(train_dur_in, val_dur_in, test_dur_in, dataset_dur)
