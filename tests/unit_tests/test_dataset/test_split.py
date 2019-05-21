import os
import unittest
from glob import glob

import numpy as np
import crowsetta

import vak.dataset.spect
import vak.dataset.annot
import vak.dataset.split
from vak.dataset.classes import VocalizationDataset
from vak.dataset.split import OnlyValDurError, InvalidDurationError, SplitsDurationGreaterThanDatasetDurationError

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.array_dir_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        self.array_list_mat = glob(os.path.join(self.array_dir_mat, '*.mat'))

        self.annot_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')
        self.scribe = crowsetta.Transcriber(voc_format='yarden')
        self.annot_list_mat = self.scribe.to_seq(self.annot_mat)
        self.labelset_mat = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

    def test_validate_durs(self):
        train_dur_in = 100
        val_dur_in = 25
        test_dur_in = 75
        vds_dur = 200
        train_dur_out, val_dur_out, test_dur_out = vak.dataset.split._validate_durs(train_dur_in,
                                                                                    val_dur_in,
                                                                                    test_dur_in,
                                                                                    vds_dur)
        self.assertTrue(
            all([out_ == in_ for out_, in_ in zip((train_dur_in, val_dur_in, test_dur_in),
                                                  (train_dur_out, val_dur_out, test_dur_out))]))

        train_dur_in = 100
        val_dur_in = None
        test_dur_in = None
        vds_dur = 200
        train_dur_out, val_dur_out, test_dur_out = vak.dataset.split._validate_durs(train_dur_in,
                                                                                    val_dur_in,
                                                                                    test_dur_in,
                                                                                    vds_dur)
        self.assertTrue(
            all([train_dur_out == train_dur_in,
                 test_dur_out == -1,
                 val_dur_out is None]))

        train_dur_in = None
        val_dur_in = None
        test_dur_in = 100
        vds_dur = 200
        train_dur_out, val_dur_out, test_dur_out = vak.dataset.split._validate_durs(train_dur_in,
                                                                                    val_dur_in,
                                                                                    test_dur_in,
                                                                                    vds_dur)
        self.assertTrue(
            all([train_dur_out == -1,
                 test_dur_out == test_dur_in,
                 val_dur_out is None]))

        train_dur_in = None
        val_dur_in = None
        test_dur_in = None
        vds_dur = 200
        with self.assertRaises(ValueError):
            # because we have to specify at least one of train_dur or test_dur
            vak.dataset.split._validate_durs(train_dur_in,
                                             val_dur_in,
                                             test_dur_in,
                                             vds_dur)

        train_dur_in = None
        val_dur_in = 100
        test_dur_in = None
        vds_dur = 200
        # because we only specified duration for validation set
        with self.assertRaises(OnlyValDurError):
            vak.dataset.split._validate_durs(train_dur_in, val_dur_in, test_dur_in, vds_dur)

        train_dur_in = -2
        test_dur_in = None
        val_dur_in = 100
        vds_dur = 200
        # because we only specified duration for validation set
        with self.assertRaises(InvalidDurationError):
            vak.dataset.split._validate_durs(train_dur_in, val_dur_in, test_dur_in, vds_dur)

        train_dur_in = 100
        test_dur_in = 100
        val_dur_in = 100
        vds_dur = 200
        # because total splits duration is greater than dataset duration
        with self.assertRaises(SplitsDurationGreaterThanDatasetDurationError):
            vak.dataset.split._validate_durs(train_dur_in, val_dur_in, test_dur_in, vds_dur)

    def test_train_test_dur_split_inds(self):
        # easy
        durs = (5, 5, 5, 5, 5)
        labelset = 'abcde'
        labels = ([np.asarray(list(labelset)) for _ in range(5)])
        train_inds, test_inds, _ = vak.dataset.split.train_test_dur_split_inds(durs,
                                                                               labels,
                                                                               labelset,
                                                                               train_dur=20,
                                                                               test_dur=5)
        all_inds_out = sorted(train_inds + test_inds)
        all_inds_in = list(range(len(durs)))
        self.assertTrue(all_inds_out == all_inds_in)

        # not as easy
        durs = (3, 2, 1, 3, 2, 3, 2, 1, 3, 2)
        labelset = 'abcde'
        labels = ['abc', 'ab', 'c', 'cde', 'de', 'abc', 'ab', 'c', 'cde', 'de']
        labels = ([np.asarray(list(lbl)) for lbl in labels])
        train_inds, test_inds, _ = vak.dataset.split.train_test_dur_split_inds(durs,
                                                                               labels,
                                                                               labelset,
                                                                               train_dur=18,
                                                                               test_dur=4)
        all_inds_out = sorted(train_inds + test_inds)
        all_inds_in = list(range(len(durs)))
        self.assertTrue(all_inds_out == all_inds_in)

        # hard
        durs = (3, 2, 1, 3, 2, 3, 2, 1, 3, 2)
        labelset = 'abcde'
        labels = ['abc', 'ab', 'c', 'cde', 'de', 'abc', 'ab', 'c', 'cde', 'de']
        labels = ([np.asarray(list(lbl)) for lbl in labels])

        with self.assertRaises(ValueError):
            vak.dataset.split.train_test_dur_split_inds(durs,
                                                        labels,
                                                        labelset,
                                                        train_dur=16,
                                                        val_dur=2,
                                                        test_dur=4)

        durs = (3, 2, 1, 3, 2, 3, 2, 1, 3, 2)
        labelset = 'abcde'
        labels = ['abc', 'ab', 'c', 'cde', 'de', 'abc', 'ab', 'c', 'cde', 'de']
        labels = ([np.asarray(list(lbl)) for lbl in labels])
        train_inds, val_inds, test_inds = vak.dataset.split.train_test_dur_split_inds(durs,
                                                                                      labels,
                                                                                      labelset,
                                                                                      train_dur=14,
                                                                                      val_dur=4,
                                                                                      test_dur=4)
        all_inds_out = sorted(train_inds + val_inds + test_inds)
        all_inds_in = list(range(len(durs)))
        self.assertTrue(all_inds_out == all_inds_in)

    def test_train_test_dur_split(self):
        vds = vak.dataset.spect.from_files(spect_format='mat',
                                           spect_dir=self.array_dir_mat,
                                           annot_list=self.annot_list_mat,
                                           load_spects=False)

        train_vds, test_vds = vak.dataset.split.train_test_dur_split(vds,
                                                                     labelset=self.labelset_mat,
                                                                     train_dur=200,
                                                                     test_dur=200)
        for vds in (train_vds, test_vds):
            self.assertTrue(type(vds) == VocalizationDataset)


if __name__ == '__main__':
    unittest.main()
