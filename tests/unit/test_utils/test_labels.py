import crowsetta
import numpy as np
import pandas as pd

import vak
import vak.files.spect
import vak.labeled_timebins


def test_to_map():
    labelset = set(list('abcde'))
    labelmap = vak.labels.to_map(labelset, map_unlabeled=False)
    assert type(labelmap) == dict
    assert len(labelmap) == len(labelset)  # because map_unlabeled=False

    labelset = set(list('abcde'))
    labelmap = vak.labels.to_map(labelset, map_unlabeled=True)
    assert type(labelmap) == dict
    assert len(labelmap) == len(labelset) + 1  # because map_unlabeled=True

    labelset = {1, 2, 3, 4, 5, 6}
    labelmap = vak.labels.to_map(labelset, map_unlabeled=False)
    assert type(labelmap) == dict
    assert len(labelmap) == len(labelset)  # because map_unlabeled=False

    labelset = {1, 2, 3, 4, 5, 6}
    labelmap = vak.labels.to_map(labelset, map_unlabeled=True)
    assert type(labelmap) == dict
    assert len(labelmap) == len(labelset) + 1  # because map_unlabeled=True


def test_to_set():
    labels1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
    labels2 = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
    labels_list = [labels1, labels2]
    labelset = vak.labels.to_set(labels_list)
    assert type(labelset) == set
    assert labelset == {1, 2, 3}


def test_has_unlabeled():
    labels_1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
    onsets_s1 = np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16])
    offsets_s1 = np.asarray([1, 3, 5, 7, 9, 11, 13, 15, 17])
    time_bins = np.arange(0, 18, 0.001)
    has_ = vak.labeled_timebins.has_unlabeled(labels_1, onsets_s1, offsets_s1, time_bins)
    assert has_

    labels_1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
    onsets_s1 = np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16])
    offsets_s1 = np.asarray([1.999, 3.999, 5.999, 7.999, 9.999, 11.999, 13.999, 15.999, 17.999])
    time_bins = np.arange(0, 18, 0.001)
    has_ = vak.labeled_timebins.has_unlabeled(labels_1, onsets_s1, offsets_s1, time_bins)
    assert has_ is False


def test_segment_lbl_tb():
    lbl_tb = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    labels, onset_inds, offset_inds = vak.labeled_timebins._segment_lbl_tb(lbl_tb)
    assert np.array_equal(labels, np.asarray([0, 1, 0]))
    assert np.array_equal(onset_inds, np.asarray([0, 4, 8]))
    assert np.array_equal(offset_inds, np.asarray([3, 7, 11]))


def test_lbl_tb_segment_inds_list():
    UNLABELED = 0

    lbl_tb = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    seg_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb=lbl_tb, unlabeled_label=UNLABELED)
    expected_seg_inds_list = [np.array([4, 5, 6, 7])]
    assert np.array_equal(seg_inds_list, expected_seg_inds_list)

    # assert works when segment is at start of lbl_tb
    lbl_tb = np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
    seg_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb=lbl_tb, unlabeled_label=UNLABELED)
    expected_seg_inds_list = [np.array([0, 1, 2, 3])]
    assert np.array_equal(seg_inds_list, expected_seg_inds_list)

    # assert works with multiple segments
    lbl_tb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0, 0])
    seg_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb=lbl_tb, unlabeled_label=UNLABELED)
    expected_seg_inds_list = [np.array([3, 4, 5]), np.array([9, 10, 11])]
    assert np.array_equal(seg_inds_list, expected_seg_inds_list)

    # assert works when a segment is at end of lbl_tb
    lbl_tb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1])
    seg_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb=lbl_tb, unlabeled_label=UNLABELED)
    expected_seg_inds_list = [np.array([3, 4, 5]), np.array([9, 10, 11])]
    assert np.array_equal(seg_inds_list, expected_seg_inds_list)


def test_remove_short_segments():
    UNLABELED = 0

    # should do nothing when a labeled segment has all the same labels
    lbl_tb = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0])
    segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
    TIMEBIN_DUR = 0.001
    MIN_SEGMENT_DUR = 0.002
    lbl_tb_tfm, segment_inds_list_out = vak.labeled_timebins.remove_short_segments(lbl_tb,
                                                                                   segment_inds_list,
                                                                                   timebin_dur=TIMEBIN_DUR,
                                                                                   min_segment_dur=MIN_SEGMENT_DUR,
                                                                                   unlabeled_label=UNLABELED)

    lbl_tb_expected = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    assert np.array_equal(lbl_tb_tfm, lbl_tb_expected)


def test_majority_vote():
    UNLABELED = 0

    # should do nothing when a labeled segment has all the same labels
    lbl_tb = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
    lbl_tb_tfm = vak.labeled_timebins.majority_vote_transform(lbl_tb, segment_inds_list)
    assert np.array_equal(lbl_tb, lbl_tb_tfm)

    lbl_tb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0, 0])
    segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
    lbl_tb_tfm = vak.labeled_timebins.majority_vote_transform(lbl_tb, segment_inds_list)

    lbl_tb_tfm_expected = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0])
    assert np.array_equal(lbl_tb_tfm, lbl_tb_tfm_expected)

    # test MajorityVote works when there is no 'unlabeled' segment at start of vector
    lbl_tb = np.asarray([1, 1, 2, 1, 0, 0, 0, 0])
    segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
    lbl_tb_tfm = vak.labeled_timebins.majority_vote_transform(lbl_tb, segment_inds_list)

    lbl_tb_tfm_expected = np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
    assert np.array_equal(lbl_tb_tfm, lbl_tb_tfm_expected)

    # test MajorityVote works when there is no 'unlabeled' segment at end of vector
    lbl_tb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1])
    segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
    lbl_tb_tfm = vak.labeled_timebins.majority_vote_transform(lbl_tb, segment_inds_list)

    lbl_tb_tfm_expected = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    assert np.array_equal(lbl_tb_tfm, lbl_tb_tfm_expected)

    # test that a tie results in lowest value class winning, default behavior of scipy.stats.mode
    lbl_tb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2])
    segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
    lbl_tb_tfm = vak.labeled_timebins.majority_vote_transform(lbl_tb, segment_inds_list)

    lbl_tb_tfm_expected = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])

    assert np.array_equal(lbl_tb_tfm, lbl_tb_tfm_expected)


def test_lbl_tb2segments_recovers_onsets_offsets_labels():
    onsets_s = np.asarray(
        [1., 3., 5., 7.]
    )
    offsets_s = np.asarray(
        [2., 4., 6., 8.]
    )
    labelset = set(list('abcd'))
    labelmap = vak.labels.to_map(labelset)

    labels = np.asarray(['a', 'b', 'c', 'd'])
    timebin_dur = 0.001
    total_dur_s = 10
    lbl_tb = np.zeros(
        (int(total_dur_s / timebin_dur),),
        dtype='int8',
    )
    for on, off, lbl in zip(onsets_s, offsets_s, labels):
        lbl_tb[int(on/timebin_dur):int(off/timebin_dur)] = labelmap[lbl]

    labels_out, onsets_s_out, offsets_s_out = vak.labeled_timebins.lbl_tb2segments(lbl_tb,
                                                                                   labelmap,
                                                                                   timebin_dur)

    assert np.array_equal(labels, labels_out)
    assert np.allclose(onsets_s, onsets_s_out, atol=0.001, rtol=0.03)
    assert np.allclose(offsets_s, offsets_s_out, atol=0.001, rtol=0.03)


def test_lbl_tb2segments_recovers_onsets_offsets_labels_from_real_data(
        config_toml_train_audio_cbin_annot_notmat
):
    """test that ``lbl_tb2segments`` recovers onsets and offsets from real data

    take all the annotations
    """
    # ---- unpack a bunch of stuff from config that we use for this test
    config = config_toml_train_audio_cbin_annot_notmat
    csv_fname = config['TRAIN']['csv_fname']
    vak_df = pd.read_csv(csv_fname)
    annot_paths = vak_df['annot_path'].values
    spect_paths = vak_df['spect_path'].values
    labelmap = vak.labels.to_map(
            set(list(config['PREP']['labelset']))
        )
    timebin_dur = vak.io.dataframe.validate_and_get_timebin_dur(vak_df)
    timebins_key = 't'

    scribe = crowsetta.Transcriber(annot_format='notmat')
    annot_list = scribe.from_file(annot_file=annot_paths)
    annot_list = [
        annot
        for annot in annot_list
        # need to remove any annotations that have labels not in labelset
        if not any(lbl not in labelmap.keys() for lbl in annot.seq.labels)
    ]
    spect_annot_map = vak.annotation.source_annot_map(
        spect_paths,
        annot_list,
    )

    lbl_tb_list = []
    for spect_file, annot in spect_annot_map.items():
        lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
        time_bins = vak.files.spect.load(spect_file)[timebins_key]
        lbl_tb_list.append(
            vak.labeled_timebins.label_timebins(lbls_int,
                                                annot.seq.onsets_s,
                                                annot.seq.offsets_s,
                                                time_bins,
                                                unlabeled_label=labelmap['unlabeled'])
        )

    for lbl_tb, annot in zip(lbl_tb_list, spect_annot_map.values()):
        labels, onsets_s, offsets_s = vak.labeled_timebins.lbl_tb2segments(lbl_tb,
                                                                           labelmap,
                                                                           timebin_dur)

        assert np.array_equal(labels, annot.seq.labels)
        assert np.allclose(onsets_s, annot.seq.onsets_s, atol=0.001, rtol=0.03)
        assert np.allclose(offsets_s, annot.seq.offsets_s, atol=0.001, rtol=0.03)


def test_lbl_tb2segments_majority_vote():
    labelmap = {
        0: 'unlabeled',
        1: 'a',
        2: 'b',
    }
    lbl_tb = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 2, 2, 1, 0, 0])
    labels_out, onsets_s_out, offsets_s_out = vak.labeled_timebins.lbl_tb2segments(lbl_tb,
                                                                                   labelmap,
                                                                                   timebin_dur=0.001,
                                                                                   majority_vote=True)
    assert labels_out == ['a', 'b']
