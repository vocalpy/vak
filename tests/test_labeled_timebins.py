import numpy as np
import pytest

import vak.files.spect
import vak.labeled_timebins


def test_has_unlabeled():
    labels_1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
    onsets_s1 = np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16])
    offsets_s1 = np.asarray([1, 3, 5, 7, 9, 11, 13, 15, 17])
    time_bins = np.arange(0, 18, 0.001)
    has_ = vak.labeled_timebins.has_unlabeled(
        labels_1, onsets_s1, offsets_s1, time_bins
    )
    assert has_

    labels_1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
    onsets_s1 = np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16])
    offsets_s1 = np.asarray(
        [1.999, 3.999, 5.999, 7.999, 9.999, 11.999, 13.999, 15.999, 17.999]
    )
    time_bins = np.arange(0, 18, 0.001)
    has_ = vak.labeled_timebins.has_unlabeled(
        labels_1, onsets_s1, offsets_s1, time_bins
    )
    assert has_ is False


@pytest.mark.parametrize(
    "labeled_timebins, labels_mapping, spect_ID_vector, expected_labels",
    [
        (np.array([0, 0, 1, 1, 0, 0, 2, 2, 0, 0]), {'unlabeled': 0, 'a': 1, 'b': 2}, None, 'ab'),
        (np.array([0, 0, 1, 1, 0, 0, 2, 2, 0, 0]), {'unlabeled': 0, '1': 1, '2': 2}, None, '12'),
        (np.array([0, 0, 21, 21, 0, 0, 22, 22, 0, 0]), {'unlabeled': 0, '21': 21, '22': 22}, None, 'AB'),
        (np.array([0, 0, 11, 11, 0, 0, 12, 12, 0, 0]), {'unlabeled': 0, '11': 11, '12': 12}, None, 'AB'),
    ]
)
def test_lbl_tb2labels(labeled_timebins, labels_mapping, spect_ID_vector, expected_labels):
    labels = vak.labeled_timebins.lbl_tb2labels(labeled_timebins, labels_mapping, spect_ID_vector)
    assert labels == expected_labels


def test_segment_lbl_tb():
    lbl_tb = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    labels, onset_inds, offset_inds = vak.labeled_timebins._segment_lbl_tb(lbl_tb)
    assert np.array_equal(labels, np.asarray([0, 1, 0]))
    assert np.array_equal(onset_inds, np.asarray([0, 4, 8]))
    assert np.array_equal(offset_inds, np.asarray([3, 7, 11]))


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
def test_lbl_tb_segment_inds_list(lbl_tb, seg_inds_list_expected):
    UNLABELED = 0

    seg_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(
        lbl_tb=lbl_tb, unlabeled_label=UNLABELED
    )
    assert np.array_equal(seg_inds_list, seg_inds_list_expected)


def test_remove_short_segments():
    UNLABELED = 0

    # should do nothing when a labeled segment has all the same labels
    lbl_tb = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0])
    segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(
        lbl_tb, unlabeled_label=UNLABELED
    )
    TIMEBIN_DUR = 0.001
    MIN_SEGMENT_DUR = 0.002
    lbl_tb_tfm, segment_inds_list_out = vak.labeled_timebins.remove_short_segments(
        lbl_tb,
        segment_inds_list,
        timebin_dur=TIMEBIN_DUR,
        min_segment_dur=MIN_SEGMENT_DUR,
        unlabeled_label=UNLABELED,
    )

    lbl_tb_expected = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    assert np.array_equal(lbl_tb_tfm, lbl_tb_expected)


@pytest.mark.parametrize(
    "lbl_tb_in, lbl_tb_expected",
    [
        # should do nothing when a labeled segment has all the same labels
        (
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0, 0]),
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]),
        ),
        # test MajorityVote works when there is no 'unlabeled' segment at start of vector
        (np.asarray([1, 1, 2, 1, 0, 0, 0, 0]), np.asarray([1, 1, 1, 1, 0, 0, 0, 0])),
        # test MajorityVote works when there is no 'unlabeled' segment at end of vector
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1]),
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),
        ),
        # test that a tie results in lowest value class winning, default behavior of scipy.stats.mode
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]),
        ),
    ],
)
def test_majority_vote(lbl_tb_in, lbl_tb_expected):
    UNLABELED = 0

    segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(
        lbl_tb_in, unlabeled_label=UNLABELED
    )
    lbl_tb_maj_vote = vak.labeled_timebins.majority_vote_transform(
        lbl_tb_in, segment_inds_list
    )
    assert np.array_equal(lbl_tb_maj_vote, lbl_tb_expected)


MAX_ABS_DIFF = 0.003  # milliseconds


def test_lbl_tb2segments_recovers_onsets_offsets_labels():
    onsets_s = np.asarray([1.0, 3.0, 5.0, 7.0])
    offsets_s = np.asarray([2.0, 4.0, 6.0, 8.0])
    labelset = set(list("abcd"))
    labelmap = vak.labels.to_map(labelset)

    labels = np.asarray(["a", "b", "c", "d"])
    timebin_dur = 0.001
    total_dur_s = 10
    timebins = (
        np.asarray(range(1, int(total_dur_s / timebin_dur) + 1)) * timebin_dur
    )  # [0.001, 0.002, ..., 10.0]
    lbl_tb = np.zeros(timebins.shape, dtype="int8")
    for onset, offset, lbl in zip(onsets_s, offsets_s, labels):
        on_ind = np.nonzero(timebins == onset)[0].item()
        off_ind = np.nonzero(timebins == offset)[0].item()
        lbl_tb[on_ind : off_ind + 1] = labelmap[lbl]

    labels_out, onsets_s_out, offsets_s_out = vak.labeled_timebins.lbl_tb2segments(
        lbl_tb, labelmap, timebins
    )

    assert np.array_equal(labels, labels_out)
    assert np.all(np.abs(onsets_s - onsets_s_out) < MAX_ABS_DIFF)
    assert np.all(np.abs(offsets_s - offsets_s_out) < MAX_ABS_DIFF)


# skip these for now because they cause tests to fail for reasons unrelated
# to what the test is testing
SPECT_FILES_TO_SKIP = [
    "llb3_0071_2018_04_23_17_38_30.wav.mat",  # has zero duration between syllable segments, onsets 54 and 55
    # I assume the same issue is coming up with these other two
    "llb3_0074_2018_04_23_17_41_08.wav.mat",
    "llb3_0016_2018_04_23_15_18_14.wav.mat",
]


def test_lbl_tb2segments_recovers_onsets_offsets_labels_from_real_data(
    specific_dataframe,
    labelset_yarden,
    model,
):
    """test that ``lbl_tb2segments`` recovers onsets and offsets from real data"""
    vak_df = specific_dataframe(
        config_type="train", model=model, spect_format="mat", annot_format="yarden"
    )
    labelmap = vak.labels.to_map(set(labelset_yarden))

    spect_paths = vak_df["spect_path"].values
    annot_list = vak.annotation.from_df(vak_df)
    spect_annot_map = vak.annotation.map_annotated_to_annot(spect_paths, annot_list)

    TIMEBINS_KEY = "t"

    for spect_path, annot in spect_annot_map.items():
        # in general not good to have conditionals in tests
        # but neglecting these weird edge case files for now
        if any(
            spect_path.endswith(spect_file_to_skip)
            for spect_file_to_skip in SPECT_FILES_TO_SKIP
        ):
            continue

        lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
        timebins = vak.files.spect.load(spect_path)[TIMEBINS_KEY]

        lbl_tb = vak.labeled_timebins.label_timebins(
            lbls_int,
            annot.seq.onsets_s,
            annot.seq.offsets_s,
            timebins,
            unlabeled_label=labelmap["unlabeled"],
        )

        labels, onsets_s_out, offsets_s_out = vak.labeled_timebins.lbl_tb2segments(
            lbl_tb, labelmap, timebins
        )
        assert np.all(np.char.equal(labels, annot.seq.labels))
        assert np.all(np.abs(annot.seq.onsets_s - onsets_s_out) < MAX_ABS_DIFF)
        assert np.all(np.abs(annot.seq.offsets_s - offsets_s_out) < MAX_ABS_DIFF)


def test_lbl_tb2segments_majority_vote():
    labelmap = {
        "unlabeled": 0,
        "a": 1,
        "b": 2,
    }
    lbl_tb = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 2, 2, 1, 0, 0])
    timebins = np.arange(1, lbl_tb.shape[0] + 1) * 0.001
    labels_out, onsets_s_out, offsets_s_out = vak.labeled_timebins.lbl_tb2segments(
        lbl_tb, labelmap, timebins, majority_vote=True
    )
    assert np.all(np.char.equal(labels_out, np.array(["a", "b"])))


def test_lbl_tb2segments_all_unlabeled():
    """test that ``lbl_tb2segments`` returns all ``None``s when
    all elements in the input vector ``lbl_tb`` are the ``unlabeled`` class"""
    labelmap = {
        "unlabeled": 0,
        "a": 1,
        "b": 2,
    }
    N_TIMEBINS = 4000  # just want some number that's on the order of size of a typical Bengalese finch song
    lbl_tb = np.zeros(N_TIMEBINS).astype(int)
    timebins = np.arange(1, lbl_tb.shape[0] + 1) * 0.001
    labels_out, onsets_s_out, offsets_s_out = vak.labeled_timebins.lbl_tb2segments(
        lbl_tb, labelmap, timebins, majority_vote=True
    )
    assert all([out is None for out in [labels_out, onsets_s_out, offsets_s_out]])


@pytest.mark.parametrize(
    'y_pred, timebin_dur, min_segment_dur, labelmap',
    [
        (np.array([0, 0, 0, 0, 0, 0, 7, 7, 3,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, ]),
         0.002,
         0.025,
         {"unlabeled": 0, "a": 3, "b": 7}),
    ]
)
def test_lbl_tb2segments_min_seg_dur_makes_all_unlabeled(y_pred,
                                                         timebin_dur,
                                                         min_segment_dur,
                                                         labelmap):
    """test that ``lbl_tb2segments`` returns all ``None``s when
    removing all segments less than the minimum segment duration
    causes all elements in the input vector ``lbl_tb``
    to become the ``unlabeled`` class"""
    # TODO: assert that applying 'minimum segment duration' post-processing does what we expect
    # i.e. converts all elements to 'unlabeled'
    timebins = np.arange(1, y_pred.shape[0] + 1) * timebin_dur
    labels_out, onsets_s_out, offsets_s_out = vak.labeled_timebins.lbl_tb2segments(
        y_pred, labelmap, timebins, min_segment_dur=min_segment_dur, majority_vote=True
    )
    assert all([out is None for out in [labels_out, onsets_s_out, offsets_s_out]])
