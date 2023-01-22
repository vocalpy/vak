"""tests for functional forms of transforms
for labeled timebins.

Tests are in the same order as the module ``vak.transforms.labeled_timebins.functional``.:
- from_segments: transform to get labeled timebins from annotations
- to_labels: transform to get back just string labels from labeled timebins,
  used to evaluate a model
- to_segments: transform to get back segment onsets, offsets, and labels from labeled timebins.
  Inverse of ``from_segments``.
- post-processing transforms that can be used to "clean up" a vector of labeled timebins
  - to_inds_list: helper function used to find segments in a vector of labeled timebins
  - remove_short_segments: remove any segment less than a minimum duration
  - take_majority_vote: take a "majority vote" within each segment bounded by the "unlabeled" label,
    and apply the most "popular" label within each segment to all timebins in that segment

Additionally some of the functions have more than one unit test,
where the first tests with simple examples
and the second then tests with real data.
Namely, ``to_labels``, ``to_segments`` and the related functions
``to_labels_with_postprocessing``
and ``to_segments_with_postprocessing``.
Simple examples are used to test expected behavior and edge cases.
Testing with real data complements this.
"""
import copy
import itertools

import numpy as np
import pytest

import vak.files.spect
import vak.labels
import vak.transforms.labeled_timebins


from ...fixtures.annot import ANNOT_LIST_YARDEN, ANNOT_LIST_NOTMAT, LABELSET_YARDEN, LABELSET_NOTMAT
from ...fixtures.spect import SPECT_LIST_NPZ, SPECT_LIST_MAT


assert len(ANNOT_LIST_YARDEN) == len(SPECT_LIST_MAT), "ANNOT_LIST_YARDEN and SPECT_LIST_MAT are not the same length"

SPECT_LIST_NPZ = copy.deepcopy(SPECT_LIST_NPZ)  # to not mutate the one used by fixtures
ANNOT_LIST_NOTMAT = copy.deepcopy(ANNOT_LIST_NOTMAT)  # to not mutate the one used by fixtures
# make sure ANNOT_LIST_NOTMAT can pair with SPECT_LIST_NPZ
audio_paths_from_spect_list = [
    spect_path.name.replace('.spect.npz', '') for spect_path in SPECT_LIST_NPZ
]
ANNOT_LIST_NOTMAT = [
    annot for annot in ANNOT_LIST_NOTMAT
    if annot.audio_path.name in audio_paths_from_spect_list
]


# define here because we re-use to parametrize multiple tests
# and because we import in .test_transforms
FROM_SEGMENTS_PARAMETRIZE_ARGVALS = list(zip(
    sorted(ANNOT_LIST_YARDEN, key=lambda annot: annot.audio_path.name),
    sorted(SPECT_LIST_MAT, key=lambda spect_path: spect_path.name),
    itertools.repeat(LABELSET_YARDEN)
)) + list(zip(
    sorted(ANNOT_LIST_NOTMAT, key=lambda annot: annot.audio_path.name),
    sorted(SPECT_LIST_NPZ, key=lambda spect_path: spect_path.name),
    itertools.repeat(LABELSET_NOTMAT)
))


@pytest.mark.parametrize(
    'annot, spect_path, labelset',
    FROM_SEGMENTS_PARAMETRIZE_ARGVALS,
)
def test_from_segments(annot, spect_path, labelset):
    labelset = vak.converters.labelset_to_set(labelset)
    labelmap = vak.labels.to_map(labelset, True)

    spect_dict = vak.files.spect.load(spect_path)
    timebins = spect_dict['t']

    try:
        lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
    except KeyError:
        pytest.skip(
            'Annotation with label not in labelset, would not include in dataset'
        )

    lbl_tb = vak.transforms.labeled_timebins.from_segments(
        lbls_int,
        annot.seq.onsets_s,
        annot.seq.offsets_s,
        timebins,
        unlabeled_label=labelmap['unlabeled'],
    )
    assert lbl_tb.shape == timebins.shape
    assert all(
        [lbl in lbls_int for lbl in np.unique(lbls_int)]
    )


@pytest.mark.parametrize(
    "lbl_tb, labelmap, labels_expected_int",
    [
        (np.array([0, 0, 1, 1, 0, 0, 2, 2, 0, 0]), {'unlabeled': 0, 'a': 1, 'b': 2}, [1, 2]),
        (np.array([0, 0, 1, 1, 0, 0, 2, 2, 0, 0]), {'unlabeled': 0, '1': 1, '2': 2}, [1, 2]),
        (np.array([0, 0, 21, 21, 0, 0, 22, 22, 0, 0]), {'unlabeled': 0, '21': 21, '22': 22}, [21, 22]),
        (np.array([0, 0, 11, 11, 0, 0, 12, 12, 0, 0]), {'unlabeled': 0, '11': 11, '12': 12}, [11, 12]),
    ]
)
def test_to_labels(lbl_tb, labelmap, labels_expected_int):
    # next line, convert all labels to single characters
    # we can easily compare strings we get back with expected;
    # this is what core.eval does
    labelmap = vak.labels.multi_char_labels_to_single_char(
        labelmap, skip=('unlabeled',)
    )
    labelmap_inv = {v: k for k, v in labelmap.items()}
    labels_expected = ''.join([labelmap_inv[lbl_int] for lbl_int in labels_expected_int])

    labels = vak.transforms.labeled_timebins.to_labels(lbl_tb, labelmap)
    assert labels == labels_expected


# skip these for now because they cause tests to fail for reasons unrelated
# to what the test is testing
SPECT_FILES_TO_SKIP = [
    "llb3_0071_2018_04_23_17_38_30.wav.mat",  # has zero duration between syllable segments, onsets 54 and 55
    # these have similar issues, where we can't successfully round trip from labeled timebins to segments
    # because the timebin duration is pretty big (2.7 ms) and there are silent gap durations very close to that
    # (e.g. 3 ms), so segments get combined or lost due to rounding error when we do np.min/max below
    "llb3_0074_2018_04_23_17_41_08.wav.mat",
    "llb3_0016_2018_04_23_15_18_14.wav.mat",
    "llb3_0053_2018_04_23_17_20_04.wav.mat",
    "llb3_0054_2018_04_23_17_21_23.wav.mat"
]


@pytest.mark.parametrize(
    'annot, spect_path, labelset',
    FROM_SEGMENTS_PARAMETRIZE_ARGVALS,
)
def test_to_labels_real_data(
        annot, spect_path, labelset
):
    """test that ``to_labels_with_postprocessing`` recovers labels from real data"""
    labelset = vak.converters.labelset_to_set(labelset)
    labelmap = vak.labels.to_map(labelset)
    # next line, convert all labels to single characters
    # we can easily compare strings we get back with expected;
    # this is what core.eval does
    labelmap = vak.labels.multi_char_labels_to_single_char(
        labelmap, skip=('unlabeled',)
    )
    TIMEBINS_KEY = "t"

    if any(
        str(spect_path).endswith(spect_file_to_skip)
        for spect_file_to_skip in SPECT_FILES_TO_SKIP
    ):
        pytest.skip(
            "Can't round trip segments -> lbl_tb -> segments "
            "because of small silent gap durations + large time bin durations"
        )

    try:
        lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
    except KeyError:
        pytest.skip(
            'Annotation with label not in labelset, would not include in dataset'
        )

    timebins = vak.files.spect.load(spect_path)[TIMEBINS_KEY]

    lbl_tb = vak.transforms.labeled_timebins.from_segments(
        lbls_int,
        annot.seq.onsets_s,
        annot.seq.offsets_s,
        timebins,
        unlabeled_label=labelmap["unlabeled"],
    )

    labels = vak.transforms.labeled_timebins.to_labels(
        lbl_tb,
        labelmap,
    )

    labelmap_multi_inv = {v: k for k, v in
                          labelmap.items()}
    labels_expected = "".join(
        [labelmap_multi_inv[lbl_int] for lbl_int in lbls_int]
    )
    assert labels == labels_expected


MAX_ABS_DIFF = 0.003  # milliseconds


@pytest.mark.parametrize(
    'annot, spect_path, labelset',
    FROM_SEGMENTS_PARAMETRIZE_ARGVALS,
)
def test_to_segments_real_data(
        annot, spect_path, labelset
):
    """test that ``to_segments`` recovers onsets, offsets, and labels from real data"""
    labelset = vak.converters.labelset_to_set(labelset)
    labelmap = vak.labels.to_map(labelset)

    TIMEBINS_KEY = "t"

    if any(
        str(spect_path).endswith(spect_file_to_skip)
        for spect_file_to_skip in SPECT_FILES_TO_SKIP
    ):
        pytest.skip(
            "Can't round trip segments -> lbl_tb -> segments "
            "because of small silent gap durations + large time bin durations"
        )

    try:
        lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
    except KeyError:
        pytest.skip(
            'Annotation with label not in labelset, would not include in dataset'
        )

    timebins = vak.files.spect.load(spect_path)[TIMEBINS_KEY]

    lbl_tb = vak.transforms.labeled_timebins.from_segments(
        lbls_int,
        annot.seq.onsets_s,
        annot.seq.offsets_s,
        timebins,
        unlabeled_label=labelmap["unlabeled"],
    )

    expected_labels = lbl_tb[np.insert(np.diff(lbl_tb).astype(bool), 0, True)]

    labels, onsets_s, offsets_s = vak.transforms.labeled_timebins.to_segments(
        lbl_tb, labelmap, timebins
    )

    assert np.all(np.char.equal(labels, annot.seq.labels))
    # writing the logic of the function here to test wouldn't make sense
    # but to still test on real data, we can test whether onset_inds
    # is the same length as expected_labels. This should be True
    assert np.all(np.abs(annot.seq.onsets_s - onsets_s) < MAX_ABS_DIFF)
    assert np.all(np.abs(annot.seq.offsets_s - offsets_s) < MAX_ABS_DIFF)


@pytest.mark.parametrize(
    "lbl_tb, seg_inds_list_expected",
    [
        (np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]), [np.array([4, 5, 6, 7])]),
        # assert works when segment is at start of lbl_tb
        (np.asarray([1, 1, 1, 1, 0, 0, 0, 0]), [np.array([0, 1, 2, 3])]),
        # assert works with multiple segments
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0, 0]),
            [np.array([3, 4, 5]), np.array([9, 10, 11])],
        ),
        # assert works when a segment is at end of lbl_tb
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1]),
            [np.array([3, 4, 5]), np.array([9, 10, 11])],
        ),
    ],
)
def test_to_inds(lbl_tb, seg_inds_list_expected):
    """Test ``to_inds`` works as expected"""
    UNLABELED = 0

    seg_inds_list = vak.transforms.labeled_timebins.to_inds_list(
        lbl_tb=lbl_tb, unlabeled_label=UNLABELED
    )
    assert np.array_equal(seg_inds_list, seg_inds_list_expected)


@pytest.mark.parametrize(
    'lbl_tb, unlabeled, timebin_dur, min_segment_dur, lbl_tb_expected',
    [
        # should remove the 1 at the end if lbl_tb since it's a segment with dur < 0.002
        (
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]),
            0,
            0.001,
            0.002,
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        ),
        # should **not** remove a segment with dur == 0.002
        (
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]),
            0,
            0.001,
            0.002,
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0])
        )
    ]
)
def test_remove_short_segments(lbl_tb, unlabeled, timebin_dur, min_segment_dur, lbl_tb_expected):
    """Test ``remove_short_segments`` works as expected"""
    segment_inds_list = vak.transforms.labeled_timebins.to_inds_list(
        lbl_tb, unlabeled_label=unlabeled
    )
    lbl_tb_tfm, segment_inds_list_out = vak.transforms.labeled_timebins.remove_short_segments(
        lbl_tb,
        segment_inds_list,
        timebin_dur=timebin_dur,
        min_segment_dur=min_segment_dur,
        unlabeled_label=unlabeled,
    )
    assert np.array_equal(lbl_tb_tfm, lbl_tb_expected)


@pytest.mark.parametrize(
    "lbl_tb_in, unlabeled, lbl_tb_expected",
    [
        # should do nothing when a labeled segment has all the same labels
        (
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
            0,
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0, 0]),
            0,
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]),
        ),
        # test MajorityVote works when there is no 'unlabeled' segment at start of vector
        (
            np.asarray([1, 1, 2, 1, 0, 0, 0, 0]),
            0,
            np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
        ),
        # test MajorityVote works when there is no 'unlabeled' segment at end of vector
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1]),
            0,
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),
        ),
        # test that a tie results in lowest value class winning, default behavior of scipy.stats.mode
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2]),
            0,
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]),
        ),
    ],
)
def test_majority_vote(lbl_tb_in, unlabeled, lbl_tb_expected):
    """Test ``majority_vote`` works as expected"""
    segment_inds_list = vak.transforms.labeled_timebins.to_inds_list(
        lbl_tb_in, unlabeled_label=unlabeled
    )
    lbl_tb_maj_vote = vak.transforms.labeled_timebins.take_majority_vote(
        lbl_tb_in, segment_inds_list
    )
    assert np.array_equal(lbl_tb_maj_vote, lbl_tb_expected)


# ---- define these constants here we use with pytest.mark.parametrize
# so that we can import them in .test_transforms as well
TIMEBIN_DUR_FOR_PARAMETRIZE = 0.001
UNLABELED_LABEL = 0
POSTPROCESS_PARAMS_ARGVALS = [
    # test case where we apply *neither* of the transforms
    (
            np.asarray([0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 0, 4, 4, 0, 0]),
            None,
            False,
            np.asarray([0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 0, 4, 4, 0, 0]),
    ),
    # test case where we apply *neither* of the transforms, and one segment is at end of lbl_tb
    (
            np.asarray([0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 0, 4, 4, 4, 4]),
            None,
            False,
            np.asarray([0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 0, 4, 4, 4, 4]),
    ),
    # ---- start of test cases for majority vote
    # test MajorityVote does nothing when a labeled segment has all the same labels
    (
        np.asarray([0, 1, 1, 0, 2, 2, 2, 2, 0, 0, 0, 0]),
        None,
        True,
        np.asarray([0, 1, 1, 0, 2, 2, 2, 2, 0, 0, 0, 0]),
    ),
    # test majority vote
    (
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0, 0]),
        None,
        True,
        # majority vote converts second segment to label "a"
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]),
    ),
    # test MajorityVote works when there is no 'unlabeled' segment at start of vector
    (
        np.array([1, 1, 2, 1, 0, 0, 0, 0]),
        None,
        True,
        np.array([1, 1, 1, 1, 0, 0, 0, 0]),
    ),
    # test MajorityVote works when there is no 'unlabeled' segment at end of vector
    (
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1]),
        None,
        True,
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),
    ),
    # test that a tie results in lowest value class winning, default behavior of scipy.stats.mode
    (
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2]),
        None,
        True,
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]),
    ),
    # test that majority vote just returns lbl_tb untouched when everything is unlabeled
    (
        np.ones(4000).astype(int) * UNLABELED_LABEL,  # i.e. all zeros, but being explicit here
        None,
        True,
        np.ones(4000).astype(int) * UNLABELED_LABEL,
    ),
    # ---- start of test cases for min segment dur
    # should remove a segment with dur < min_segment_dur
    (
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]),
        0.002,
        False,
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
    ),
    # should **not** remove a segment with dur == 0.002
    (
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]),
        0.002,
        False,
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]),
    ),
    # test min_segment_dur returns all Nones when all segments are less than min segment dur
    (
        np.array([0, 0, 0, 0, 0, 0, 1, 1, 2,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ]),
        0.025,  # notice segment dur, 25ms. Realistic value but will remove all segments in lbl_tb
        False,
        np.ones(36).astype(int) * UNLABELED_LABEL,  # i.e. all zeros, but being explicit here
    ),
]

# now rewrite but with args in order for function call:
POSTPROCESS_PARAMS_ARGVALS = [
    argvals[:1] + (TIMEBIN_DUR_FOR_PARAMETRIZE, UNLABELED_LABEL) + argvals[1:]
    for argvals in POSTPROCESS_PARAMS_ARGVALS
]


@pytest.mark.parametrize(
    'lbl_tb, timebin_dur, unlabeled_label, min_segment_dur, majority_vote, lbl_tb_expected',
    POSTPROCESS_PARAMS_ARGVALS
)
def test_postprocess(lbl_tb, timebin_dur, unlabeled_label, min_segment_dur, majority_vote, lbl_tb_expected):
    """Test that ``trasnforms.labeled_timebins.postprocess`` works as expected.
    Specifically test that we recover an expected string of labels,
    as would be used to compute edit distance."""
    lbl_tb = vak.transforms.labeled_timebins.postprocess(
        lbl_tb,
        timebin_dur=timebin_dur,
        unlabeled_label=UNLABELED_LABEL,
        majority_vote=majority_vote,
        min_segment_dur=min_segment_dur,
    )

    assert np.all(np.equal(lbl_tb, lbl_tb_expected))
