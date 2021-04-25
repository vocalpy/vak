import pytest

import vak.metrics.distance.functional


LEV_PARAMETRIZE = [
    # adapted from https://github.com/toastdriven/pylev/blob/master/tests.py
    ("kitten", "sitting", 3),
    ("kitten", "kitten", 0),
    ("", "", 0),
    ("kitten", "", 6),
    ("", "sitting", 7),
    ("meilenstein", "levenshtein", 4),
    ("levenshtein", "frankenstein", 6),
    ("confide", "deceit", 6),
    ("CUNsperrICY", "conspiracy", 8),
    # case added to catch failure with our previous implementation from
    # https://en.wikibooks.org/wiki/Talk:Algorithm_Implementation/Strings/Levenshtein_distance#Bug_in_vectorized_(5th)_version
    ('aabcc', 'bccdd', 4),
]


@pytest.mark.parametrize(
    "source, target, expected_distance",
    LEV_PARAMETRIZE
)
def test_levenshtein(source, target, expected_distance):
    distance = vak.metrics.distance.functional.levenshtein(source, target)
    assert distance == expected_distance


@pytest.mark.parametrize(
    "y_pred, y_true, expected_distance",
    LEV_PARAMETRIZE
)
def test_segment_error_rate(y_pred, y_true, expected_distance):
    if len(y_true) == 0 and len(y_pred) == 0:
        expected_rate = 0
        rate = vak.metrics.distance.functional.segment_error_rate(y_pred, y_true)
        assert rate == expected_rate
    elif len(y_true) == 0 and len(y_pred) != 0:
        with pytest.raises(ValueError):
            vak.metrics.distance.functional.segment_error_rate(y_pred, y_true)
    else:
        expected_rate = expected_distance / len(y_true)
        rate = vak.metrics.distance.functional.segment_error_rate(y_pred, y_true)
        assert rate == expected_rate
