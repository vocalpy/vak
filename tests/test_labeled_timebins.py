import numpy as np
import pytest

import vak.files.spect
import vak.labeled_timebins


@pytest.mark.parametrize(
    'labels, onsets, offsets, time_bins, expected_output',
    [
        (
            [1, 1, 1, 1, 2, 2, 3, 3, 3],
            np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16]),
            np.asarray([1, 3, 5, 7, 9, 11, 13, 15, 17]),
            np.arange(0, 18, 0.001),
            True
        ),
        (
            [1, 1, 1, 1, 2, 2, 3, 3, 3],
            np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16]),
            np.asarray([1.999, 3.999, 5.999, 7.999, 9.999, 11.999, 13.999, 15.999, 17.999]),
            np.arange(0, 18, 0.001),
            False
        )
    ],
)
def test_has_unlabeled(labels, onsets, offsets, time_bins, expected_output):
    assert vak.labeled_timebins.has_unlabeled(
        labels, onsets, offsets, time_bins
    ) == expected_output
