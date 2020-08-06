from configparser import ConfigParser
from pathlib import Path
import unittest

import crowsetta
import numpy as np
import pandas as pd

import vak
import vak.files.spect
import vak.labeled_timebins

HERE = Path(__file__).parent

# TODO: this should become a fixture when switching to PyTest!!!
PROJECT_ROOT = HERE.joinpath('..', '..', '..')

SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')
TMP_PREP_TRAIN_CONFIG = SETUP_SCRIPTS_DIR.joinpath('tmp_prep_train_config.ini')
if not TMP_PREP_TRAIN_CONFIG.exists():
    raise FileNotFoundError(
        f'config file not found in setup_scripts directory that is needed for tests: {TMP_PREP_TRAIN_CONFIG}'
    )
CONFIG = ConfigParser()
CONFIG.read(TMP_PREP_TRAIN_CONFIG)
CSV_FNAME = CONFIG['TRAIN']['csv_fname']
VAK_DF = pd.read_csv(CSV_FNAME)
ANNOT_PATHS = VAK_DF['annot_path'].values
SPECT_PATHS = VAK_DF['spect_path'].values
LABELMAP = vak.labels.to_map(
    set(list(CONFIG['PREP']['labelset']))
)
TIMEBIN_DUR = vak.io.dataframe.validate_and_get_timebin_dur(VAK_DF)
TIMEBINS_KEY = 't'


class TestLabels(unittest.TestCase):
    def test_to_map(self):
        labelset = set(list('abcde'))
        labelmap = vak.labels.to_map(labelset, map_unlabeled=False)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset)  # because map_unlabeled=False
        )

        labelset = set(list('abcde'))
        labelmap = vak.labels.to_map(labelset, map_unlabeled=True)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset) + 1  # because map_unlabeled=True
        )

        labelset = {1, 2, 3, 4, 5, 6}
        labelmap = vak.labels.to_map(labelset, map_unlabeled=False)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset)  # because map_unlabeled=False
        )

        labelset = {1, 2, 3, 4, 5, 6}
        labelmap = vak.labels.to_map(labelset, map_unlabeled=True)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset) + 1  # because map_unlabeled=True
        )

    def test_to_set(self):
        labels1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
        labels2 = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
        labels_list = [labels1, labels2]
        labelset = vak.labels.to_set(labels_list)
        self.assertTrue(
            type(labelset) == set
        )
        self.assertTrue(
            labelset == {1, 2, 3}
        )

    def test_has_unlabeled(self):
        labels_1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
        onsets_s1 = np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16])
        offsets_s1 = np.asarray([1, 3, 5, 7, 9, 11, 13, 15, 17])
        time_bins = np.arange(0, 18, 0.001)
        has_ = vak.labeled_timebins.has_unlabeled(labels_1, onsets_s1, offsets_s1, time_bins)
        self.assertTrue(has_ is True)

        labels_1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
        onsets_s1 = np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16])
        offsets_s1 = np.asarray([1.999, 3.999, 5.999, 7.999, 9.999, 11.999, 13.999, 15.999, 17.999])
        time_bins = np.arange(0, 18, 0.001)
        has_ = vak.labeled_timebins.has_unlabeled(labels_1, onsets_s1, offsets_s1, time_bins)
        self.assertTrue(has_ is False)

    def test_segment_lbl_tb(self):
        lbl_tb = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        labels, onset_inds, offset_inds = vak.labeled_timebins._segment_lbl_tb(lbl_tb)
        self.assertTrue(
            np.array_equal(labels, np.asarray([0, 1, 0]))
        )
        self.assertTrue(
            np.array_equal(onset_inds, np.asarray([0, 4, 8]))
        )
        self.assertTrue(
            np.array_equal(offset_inds, np.asarray([3, 7, 11]))
        )

    def test_lbl_tb_segment_inds_list(self):
        UNLABELED = 0

        lbl_tb = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        seg_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb=lbl_tb, unlabeled_label=UNLABELED)
        expected_seg_inds_list = [np.array([4, 5, 6, 7])]
        self.assertTrue(
            np.array_equal(seg_inds_list, expected_seg_inds_list)
        )

        # assert works when segment is at start of lbl_tb
        lbl_tb = np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
        seg_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb=lbl_tb, unlabeled_label=UNLABELED)
        expected_seg_inds_list = [np.array([0, 1, 2, 3])]
        self.assertTrue(
            np.array_equal(seg_inds_list, expected_seg_inds_list)
        )

        # assert works with multiple segments
        lbl_tb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0, 0])
        seg_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb=lbl_tb, unlabeled_label=UNLABELED)
        expected_seg_inds_list = [np.array([3, 4, 5]), np.array([9, 10, 11])]
        self.assertTrue(
            np.array_equal(seg_inds_list, expected_seg_inds_list)
        )

        # assert works when a segment is at end of lbl_tb
        lbl_tb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1])
        seg_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb=lbl_tb, unlabeled_label=UNLABELED)
        expected_seg_inds_list = [np.array([3, 4, 5]), np.array([9, 10, 11])]
        self.assertTrue(
            np.array_equal(seg_inds_list, expected_seg_inds_list)
        )

    def test_remove_short_segments(self):
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
        self.assertTrue(
            np.array_equal(lbl_tb_tfm, lbl_tb_expected)
        )

    def test_majority_vote(self):
        UNLABELED = 0

        # should do nothing when a labeled segment has all the same labels
        lbl_tb = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
        lbl_tb_tfm = vak.labeled_timebins.majority_vote_transform(lbl_tb, segment_inds_list)
        self.assertTrue(
            np.array_equal(lbl_tb, lbl_tb_tfm)
        )

        lbl_tb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0, 0])
        segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
        lbl_tb_tfm = vak.labeled_timebins.majority_vote_transform(lbl_tb, segment_inds_list)

        lbl_tb_tfm_expected = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        self.assertTrue(
            np.array_equal(lbl_tb_tfm, lbl_tb_tfm_expected)
        )

        # test MajorityVote works when there is no 'unlabeled' segment at start of vector
        lbl_tb = np.asarray([1, 1, 2, 1, 0, 0, 0, 0])
        segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
        lbl_tb_tfm = vak.labeled_timebins.majority_vote_transform(lbl_tb, segment_inds_list)

        lbl_tb_tfm_expected = np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
        self.assertTrue(
            np.array_equal(lbl_tb_tfm, lbl_tb_tfm_expected)
        )

        # test MajorityVote works when there is no 'unlabeled' segment at end of vector
        lbl_tb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1])
        segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
        lbl_tb_tfm = vak.labeled_timebins.majority_vote_transform(lbl_tb, segment_inds_list)

        lbl_tb_tfm_expected = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        self.assertTrue(
            np.array_equal(lbl_tb_tfm, lbl_tb_tfm_expected)
        )

        # test that a tie results in lowest value class winning, default behavior of scipy.stats.mode
        lbl_tb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2])
        segment_inds_list = vak.labeled_timebins.lbl_tb_segment_inds_list(lbl_tb, unlabeled_label=UNLABELED)
        lbl_tb_tfm = vak.labeled_timebins.majority_vote_transform(lbl_tb, segment_inds_list)

        lbl_tb_tfm_expected = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        lbl_tb_tfm = transform(lbl_tb)
        self.assertTrue(
            np.array_equal(lbl_tb_tfm, lbl_tb_tfm_expected)
        )

    def test_lbl_tb2segments_recovers_onsets_offsets_labels(self):
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

        self.assertTrue(
            np.array_equal(labels, labels_out)
        )
        self.assertTrue(
            np.allclose(onsets_s, onsets_s_out, atol=0.001, rtol=0.03)
        )
        self.assertTrue(
            np.allclose(offsets_s, offsets_s_out, atol=0.001, rtol=0.03)
        )

    def test_lbl_tb2segments_recovers_onsets_offsets_labels_from_real_data(self):
        # TODO: make all this into fixture(s?) when switching to PyTest
        scribe = crowsetta.Transcriber(annot_format='notmat')
        annot_list = scribe.from_file(annot_file=ANNOT_PATHS)
        annot_list = [
            annot
            for annot in annot_list
            # need to remove any annotations that have labels not in labelset
            if not any(lbl not in LABELMAP.keys() for lbl in annot.seq.labels)
        ]
        spect_annot_map = vak.annotation.source_annot_map(
            SPECT_PATHS,
            annot_list,
        )

        lbl_tb_list = []
        for spect_file, annot in spect_annot_map.items():
            lbls_int = [LABELMAP[lbl] for lbl in annot.seq.labels]
            time_bins = vak.files.spect.load(spect_file)[TIMEBINS_KEY]
            lbl_tb_list.append(
                vak.labeled_timebins.label_timebins(lbls_int,
                                                    annot.seq.onsets_s,
                                                    annot.seq.offsets_s,
                                                    time_bins,
                                                    unlabeled_label=LABELMAP['unlabeled'])
            )

        for lbl_tb, annot in zip(lbl_tb_list, spect_annot_map.values()):
            labels, onsets_s, offsets_s = vak.labeled_timebins.lbl_tb2segments(lbl_tb,
                                                                               LABELMAP,
                                                                               TIMEBIN_DUR)

            self.assertTrue(
                np.array_equal(labels, annot.seq.labels)
            )
            self.assertTrue(
                np.allclose(onsets_s, annot.seq.onsets_s, atol=0.001, rtol=0.03)
            )
            self.assertTrue(
                np.allclose(offsets_s, annot.seq.offsets_s, atol=0.001, rtol=0.03)
            )

    def test_lbl_tb2segments_majority_vote(self):
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
        self.assertTrue(
            labels_out == ['a', 'b']
        )


if __name__ == '__main__':
    unittest.main()
