import copy

import numpy as np
import pandas as pd
import pytest

import vak.common  # for constants
import vak.common.files.spect
import vak.common.labels


@pytest.mark.parametrize(
    'labelset, map_background',
    [
        (
            set(list("abcde")),
            True
        ),
        (
            set(list("abcde")),
            False
        ),
        (
            {1, 2, 3, 4, 5, 6},
            True,
        ),
        (
            {1, 2, 3, 4, 5, 6},
            False,
        )
    ]
)
def test_to_map(labelset, map_background):
    labelmap = vak.common.labels.to_map(labelset, map_background=map_background)
    assert isinstance(labelmap, dict)
    if map_background:
        # because map_background=True
        assert len(labelmap) == len(labelset) + 1
    else:
        # because map_background=False
        assert len(labelmap) == len(labelset)


@pytest.mark.parametrize(
    'labels_list, expected_labelset',
    [
        (
            [
                [1, 1, 1, 1, 2, 2, 3, 3, 3],
                [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
            ],
            {1, 2, 3}
        )
    ]
)
def test_to_set(labels_list, expected_labelset):
    labelset = vak.common.labels.to_set(labels_list)
    assert isinstance(labelset, set)
    assert labelset == expected_labelset


@pytest.mark.parametrize(
    'config_type, model_name, audio_format, spect_format, annot_format',
    [
        ('train', 'TweetyNet', 'cbin', None, 'notmat'),
        ('train', 'TweetyNet', None, 'mat', 'yarden'),
    ]
)
def test_from_df(config_type, model_name, audio_format, spect_format, annot_format,
                 specific_dataset_path, specific_dataset_df):
    df = specific_dataset_df(config_type, model_name, annot_format, audio_format, spect_format)
    dataset_path = specific_dataset_path(config_type, model_name, annot_format, audio_format, spect_format)

    out = vak.common.labels.from_df(df, dataset_path)
    assert isinstance(out, list)
    assert all([isinstance(labels, (np.ndarray, pd.arrays.StringArray)) for labels in out])


INTS_LABELMAP = {str(val): val for val in range(1, 20)}
INTS_LABELMAP_WITH_UNLABELED = copy.deepcopy(INTS_LABELMAP)
INTS_LABELMAP_WITH_UNLABELED[vak.common.constants.DEFAULT_BACKGROUND_LABEL] = 0

DEFAULT_SKIP = (vak.common.constants.DEFAULT_BACKGROUND_LABEL,)


@pytest.mark.parametrize(
    'labelmap, skip',
    [
        ({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, None),
        ({vak.common.constants.DEFAULT_BACKGROUND_LABEL: 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, None),
        ({vak.common.constants.DEFAULT_BACKGROUND_LABEL: 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, (vak.common.constants.DEFAULT_BACKGROUND_LABEL,)),
        (INTS_LABELMAP, None),
        (INTS_LABELMAP_WITH_UNLABELED, (vak.common.constants.DEFAULT_BACKGROUND_LABEL,))
    ]
)
def test_multi_char_labels_to_single_char(labelmap, skip):
    if skip:
        out = vak.common.labels.multi_char_labels_to_single_char(labelmap, skip)
    else:
        # test default skip
        out = vak.common.labels.multi_char_labels_to_single_char(labelmap)

    if skip:
        for skiplabel in skip:
            assert skiplabel in out
        assert all(
            [len(label) == 1
             for label in out.keys()
             if label not in skip]
        )
    else:
        assert all([
            len(label) == 1
            for label in out.keys()
            if label not in DEFAULT_SKIP
        ])
