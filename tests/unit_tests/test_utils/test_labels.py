from configparser import ConfigParser
from pathlib import Path
import unittest

import crowsetta
import numpy as np
import pandas as pd

import vak

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
LABELMAP = vak.utils.labels.to_map(
    set(list(CONFIG['PREP']['labelset']))
)
TIMEBIN_DUR = vak.io.dataframe.validate_and_get_timebin_dur(VAK_DF)
TIMEBINS_KEY = 't'


class TestLabels(unittest.TestCase):
    def test_where(self):
        labels_arr_0 = np.zeros(shape=(10,), dtype=np.int64)
        labels_arr_1 = np.ones(shape=(10,), dtype=np.int64)
        label_arrs = [labels_arr_0, labels_arr_1]
        where_in_labels = vak.utils.labels.where(label_arrs)

        self.assertTrue(list(where_in_labels.keys()) == [0, 1])
        self.assertTrue(where_in_labels[0] == np.asarray(0))
        self.assertTrue(where_in_labels[1] == np.asarray(1))

    def test_where_find_in_arr(self):
        labels_arr_0 = np.zeros(shape=(10,), dtype=np.int64)
        labels_arr_1 = np.ones(shape=(10,), dtype=np.int64)
        labels_arr_2 = np.zeros(shape=(10,), dtype=np.int64)
        labels_arr_3 = np.ones(shape=(10,), dtype=np.int64)
        label_arrs = [labels_arr_0, labels_arr_1, labels_arr_2, labels_arr_3]
        where_in_labels, where_in_arr = vak.utils.labels.where(label_arrs, find_in_arr=True)

        self.assertTrue(list(where_in_labels.keys()) == [0, 1])
        self.assertTrue(
            np.array_equal(where_in_labels[0], np.asarray([0, 2]))
        )
        self.assertTrue(
            np.array_equal(where_in_labels[1], np.asarray([1, 3]))
        )

        self.assertTrue(all([type(v) == dict] for v in where_in_arr.values()))

        self.assertTrue(
            list(where_in_arr[0].keys()) == [0, 2]
        )
        self.assertTrue(
            all(
                [
                    np.array_equal(arr, np.arange(10)) for arr in list(where_in_arr[0].values())
                ]
            )
        )
        self.assertTrue(
            list(where_in_arr[1].keys()) == [1, 3]
        )
        self.assertTrue(
            all(
                [
                    np.array_equal(arr, np.arange(10)) for arr in list(where_in_arr[1].values())
                ]
            )
        )

    def test_sort(self):
        uniq_labels = list('abc')
        occurrences = [
            np.arange(4),
            np.arange(2),
            np.arange(10),
        ]

        uniq_sorted, occur_sorted = vak.utils.labels.sort(uniq_labels, occurrences)
        occur_sorted_expected = [
            np.arange(2),
            np.arange(4),
            np.arange(10),
        ]
        self.assertTrue(uniq_sorted == ['b', 'a', 'c'])
        self.assertTrue(
            all(
                [np.array_equal(arr1, arr2) for arr1, arr2 in zip(occur_sorted, occur_sorted_expected)]
            )
        )

    def test_to_map(self):
        labelset = set(list('abcde'))
        labelmap = vak.utils.labels.to_map(labelset, map_unlabeled=False)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset)  # because map_unlabeled=False
        )

        labelset = set(list('abcde'))
        labelmap = vak.utils.labels.to_map(labelset, map_unlabeled=True)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset) + 1  # because map_unlabeled=True
        )

        labelset = {1, 2, 3, 4, 5, 6}
        labelmap = vak.utils.labels.to_map(labelset, map_unlabeled=False)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset)  # because map_unlabeled=False
        )

        labelset = {1, 2, 3, 4, 5, 6}
        labelmap = vak.utils.labels.to_map(labelset, map_unlabeled=True)
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
        labelset = vak.utils.labels.to_set(labels_list)
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
        has_ = vak.utils.labels.has_unlabeled(labels_1, onsets_s1, offsets_s1, time_bins)
        self.assertTrue(has_ is True)

        labels_1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
        onsets_s1 = np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16])
        offsets_s1 = np.asarray([1.999, 3.999, 5.999, 7.999, 9.999, 11.999, 13.999, 15.999, 17.999])
        time_bins = np.arange(0, 18, 0.001)
        has_ = vak.utils.labels.has_unlabeled(labels_1, onsets_s1, offsets_s1, time_bins)
        self.assertTrue(has_ is False)

    def test_segment_lbl_tb(self):
        lbl_tb = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        labels, onset_inds, offset_inds = vak.utils.labels._segment_lbl_tb(lbl_tb)
        self.assertTrue(
            np.array_equal(labels, np.asarray([0, 1, 0]))
        )
        self.assertTrue(
            np.array_equal(onset_inds, np.asarray([0, 4, 8]))
        )
        self.assertTrue(
            np.array_equal(offset_inds, np.asarray([3, 7, 11]))
        )

    def test_lbl_tb2segments_recovers_onsets_offsets_labels(self):
        onsets_s = np.asarray(
            [1., 3., 5., 7.]
        )
        offsets_s = np.asarray(
            [2., 4., 6., 8.]
        )
        labelset = set(list('abcd'))
        labelmap = vak.utils.labels.to_map(labelset)

        labels = np.asarray(['a', 'b', 'c', 'd'])
        timebin_dur = 0.001
        total_dur_s = 10
        lbl_tb = np.zeros(
            (int(total_dur_s / timebin_dur),),
            dtype='int8',
        )
        for on, off, lbl in zip(onsets_s, offsets_s, labels):
            lbl_tb[int(on/timebin_dur):int(off/timebin_dur)] = labelmap[lbl]

        labels_out, onsets_s_out, offsets_s_out = vak.utils.labels.lbl_tb2segments(lbl_tb,
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
        spect_annot_map = vak.io.annotation.source_annot_map(
            SPECT_PATHS,
            annot_list,
        )

        lbl_tb_list = []
        for spect_file, annot in spect_annot_map.items():
            lbls_int = [LABELMAP[lbl] for lbl in annot.seq.labels]
            time_bins = vak.io.spect.array_dict_from_path(spect_file)[TIMEBINS_KEY]
            lbl_tb_list.append(
                vak.utils.labels.label_timebins(lbls_int,
                                                annot.seq.onsets_s,
                                                annot.seq.offsets_s,
                                                time_bins,
                                                unlabeled_label=LABELMAP['unlabeled'])
            )

        for lbl_tb, annot in zip(lbl_tb_list, spect_annot_map.values()):
            labels, onsets_s, offsets_s = vak.utils.labels.lbl_tb2segments(lbl_tb,
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


if __name__ == '__main__':
    unittest.main()
