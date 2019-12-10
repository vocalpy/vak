import os
import unittest
from glob import glob
from math import isclose

import numpy as np
import crowsetta
from scipy.io import loadmat

import vak.io.dataframe
import vak.io.annotation
import vak.io.split
from vak.evfuncs import load_cbin
from vak.io.annotation import files_from_dir
from vak.utils.general import timebin_dur_from_vec
from vak.io.classes import Dataset
from vak.io.utils import OnlyValDurError, InvalidDurationError, SplitsDurationGreaterThanDatasetDurationError

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')

NUM_SAMPLES = 10  # number of times to sample behavior of random-number generator

audio_dir_cbin = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
audio_files_cbin = glob(os.path.join(audio_dir_cbin, '*.cbin'))
annot_files_cbin = files_from_dir(annot_dir=audio_dir_cbin, annot_format='notmat')
scribe_cbin = crowsetta.Transcriber(voc_format='notmat')
annot_list_cbin = scribe_cbin.to_seq(file=annot_files_cbin)
labelset_cbin = set(list('iabcdefghjk'))
durs_cbin = []
labels_cbin = []
for audio_file, annot in zip(audio_files_cbin, annot_list_cbin):
    if set(annot.labels).issubset(labelset_cbin):
        labels_cbin.append(annot.labels)
        fs, data = load_cbin(audio_file)
        durs_cbin.append(data.shape[0] / fs)

spect_dir_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
spect_files_mat = glob(os.path.join(spect_dir_mat, '*.mat'))
annot_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')
scribe_yarden = crowsetta.Transcriber(voc_format='yarden')
annot_list_mat = scribe_yarden.to_seq(annot_mat)
labelset_mat = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}
durs_mat = []
labels_mat = []
for spect_file_mat, annot in zip(spect_files_mat, annot_list_mat):
    if set(annot.labels).issubset(labelset_mat):
        labels_mat.append(annot.labels)
        mat_dict = loadmat(spect_file_mat)
        timebin_dur = timebin_dur_from_vec(mat_dict['t'])
        dur = mat_dict['s'].shape[-1] * timebin_dur
        durs_mat.append(dur)


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.labelset_cbin = labelset_cbin
        self.labels_cbin = labels_cbin
        self.durs_cbin = durs_cbin

        self.labelset_mat = labelset_mat
        self.labels_mat = labels_mat
        self.durs_mat = durs_mat

    def _check_output(self,
                      train_dur,
                      val_dur,
                      test_dur,
                      labelset,
                      durs,
                      labels,
                      train_inds,
                      val_inds,
                      test_inds):
        for split, dur_in, inds in zip(
                ('train', 'val', 'test'),
                (train_dur, val_dur, test_dur),
                (train_inds, val_inds, test_inds)):
            if dur_in is not None:
                dur_out = sum([durs[ind] for ind in inds])
                if dur_in >= 0:
                    self.assertTrue(dur_out >= dur_in)
                elif dur_in == -1:
                    if split == 'train':
                        self.assertTrue(
                            isclose(dur_out,
                                    sum(durs) - sum([durs[ind] for ind in test_inds])
                                    )
                        )
                    elif split == 'test':
                        self.assertTrue(
                            isclose(dur_out,
                            sum(durs) - sum([durs[ind] for ind in train_inds])
                                    )
                        )

                all_lbls_this_set = [lbl for ind in inds for lbl in labels[ind]]
                self.assertTrue(labelset == set(all_lbls_this_set))
            else:
                self.assertTrue(inds is None)

        self.assertTrue(set(train_inds).isdisjoint(set(test_inds)))
        if val_dur is not None:
            self.assertTrue(set(train_inds).isdisjoint(set(val_inds)))
            self.assertTrue(set(test_inds).isdisjoint(set(val_inds)))

        return True

    def test_train_test_dur_split_inds_mock_easy(self):
        durs = (5, 5, 5, 5, 5)
        labelset = set(list('abcde'))
        train_dur = 20
        val_dur = None
        test_dur = 5
        labels = ([np.asarray(list(labelset)) for _ in range(5)])
        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = vak.io.split.train_test_dur_split_inds(durs,
                                                                                     labels,
                                                                                     labelset,
                                                                                     train_dur,
                                                                                     test_dur)

        self.assertTrue(
            self._check_output(train_dur,
                               val_dur,
                               test_dur,
                               labelset,
                               durs,
                               labels,
                               train_inds,
                               val_inds,
                               test_inds))

    def test_train_test_dur_split_inds_mock_not_as_easy(self):
        durs = (3, 2, 1, 3, 2, 3, 2, 1, 3, 2)
        labelset = set(list('abcde'))
        labels = ['abc', 'ab', 'c', 'cde', 'de', 'abc', 'ab', 'c', 'cde', 'de']
        train_dur = 14
        val_dur = None
        test_dur = 8
        labels = ([np.asarray(list(lbl)) for lbl in labels])

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = vak.io.split.train_test_dur_split_inds(durs,
                                                                                     labels,
                                                                                     labelset,
                                                                                     train_dur,
                                                                                     test_dur)

            self.assertTrue(
                self._check_output(train_dur,
                                   val_dur,
                                   test_dur,
                                   labelset,
                                   durs,
                                   labels,
                                   train_inds,
                                   val_inds,
                                   test_inds))

    def test_train_test_dur_split_inds_mock_hard(self):
        durs = (3, 2, 1, 3, 2, 3, 2, 1, 3, 2)
        labelset = set(list('abcde'))
        labels = ['abc', 'ab', 'c', 'cde', 'de', 'abc', 'ab', 'c', 'cde', 'de']
        labels = ([np.asarray(list(lbl)) for lbl in labels])
        train_dur = 8
        val_dur = 7
        test_dur = 7

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = vak.io.split.train_test_dur_split_inds(durs,
                                                                                     labels,
                                                                                     labelset,
                                                                                     train_dur,
                                                                                     test_dur,
                                                                                     val_dur)

            self.assertTrue(
                self._check_output(train_dur,
                                   val_dur,
                                   test_dur,
                                   labelset,
                                   durs,
                                   labels,
                                   train_inds,
                                   val_inds,
                                   test_inds))

    def test_train_test_dur_split_inds_mock_impossible(self):
        durs = (3, 2, 1, 3, 2, 3, 2, 1, 3, 2)
        labelset = set(list('abcde'))
        labels = ['abc', 'ab', 'c', 'cde', 'de', 'abc', 'ab', 'c', 'cde', 'de']
        labels = ([np.asarray(list(lbl)) for lbl in labels])
        train_dur = 16
        val_dur = 2
        test_dur = 4
        with self.assertRaises(ValueError):
            vak.io.split.train_test_dur_split_inds(durs,
                                                   labels,
                                                   labelset,
                                                   train_dur,
                                                   test_dur,
                                                   val_dur)

    def test_train_test_dur_split_inds_cbin_train_test_val(self):
        train_dur = 35
        val_dur = 20
        test_dur = 35

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = vak.io.split.train_test_dur_split_inds(self.durs_cbin,
                                                                                     self.labels_cbin,
                                                                                     self.labelset_cbin,
                                                                                     train_dur,
                                                                                     test_dur,
                                                                                     val_dur)

            self.assertTrue(
                self._check_output(train_dur,
                                   val_dur,
                                   test_dur,
                                   self.labelset_cbin,
                                   self.durs_cbin,
                                   self.labels_cbin,
                                   train_inds,
                                   val_inds,
                                   test_inds))

    def test_train_test_dur_split_inds_cbin_train(self):
        train_dur = 35
        val_dur = None
        test_dur = -1

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = vak.io.split.train_test_dur_split_inds(self.durs_cbin,
                                                                                     self.labels_cbin,
                                                                                     self.labelset_cbin,
                                                                                     train_dur,
                                                                                     test_dur,
                                                                                     val_dur)

            self.assertTrue(
                self._check_output(train_dur,
                                   val_dur,
                                   test_dur,
                                   self.labelset_cbin,
                                   self.durs_cbin,
                                   self.labels_cbin,
                                   train_inds,
                                   val_inds,
                                   test_inds))

    def test_train_test_dur_split_inds_cbin_test(self):
        train_dur = -1
        val_dur = None
        test_dur = 35

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = vak.io.split.train_test_dur_split_inds(self.durs_cbin,
                                                                                     self.labels_cbin,
                                                                                     self.labelset_cbin,
                                                                                     train_dur,
                                                                                     test_dur,
                                                                                     val_dur)

            self.assertTrue(
                self._check_output(train_dur,
                                   val_dur,
                                   test_dur,
                                   self.labelset_cbin,
                                   self.durs_cbin,
                                   self.labels_cbin,
                                   train_inds,
                                   val_inds,
                                   test_inds))

    def test_train_test_dur_split_inds_mat_train_test_val(self):
        train_dur = 200
        val_dur = 100
        test_dur = 200

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = vak.io.split.train_test_dur_split_inds(self.durs_mat,
                                                                                     self.labels_mat,
                                                                                     self.labelset_mat,
                                                                                     train_dur,
                                                                                     test_dur,
                                                                                     val_dur)

            self.assertTrue(
                self._check_output(train_dur,
                                   val_dur,
                                   test_dur,
                                   self.labelset_mat,
                                   self.durs_mat,
                                   self.labels_mat,
                                   train_inds,
                                   val_inds,
                                   test_inds))

    def test_train_test_dur_split_inds_mat_train(self):
        train_dur = 200
        val_dur = None
        test_dur = -1

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = vak.io.split.train_test_dur_split_inds(self.durs_mat,
                                                                                     self.labels_mat,
                                                                                     self.labelset_mat,
                                                                                     train_dur,
                                                                                     test_dur,
                                                                                     val_dur)

            self.assertTrue(
                self._check_output(train_dur,
                                   val_dur,
                                   test_dur,
                                   self.labelset_mat,
                                   self.durs_mat,
                                   self.labels_mat,
                                   train_inds,
                                   val_inds,
                                   test_inds))

    def test_train_test_dur_split_inds_mat_test(self):
        train_dur = -1
        val_dur = None
        test_dur = 200

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = vak.io.split.train_test_dur_split_inds(self.durs_mat,
                                                                                     self.labels_mat,
                                                                                     self.labelset_mat,
                                                                                     train_dur,
                                                                                     test_dur,
                                                                                     val_dur)

            self.assertTrue(
                self._check_output(train_dur,
                                   val_dur,
                                   test_dur,
                                   self.labelset_mat,
                                   self.durs_mat,
                                   self.labels_mat,
                                   train_inds,
                                   val_inds,
                                   test_inds))

    def test_train_test_dur_split_None_raises(self):
        durs = (5, 5, 5, 5, 5)
        labelset = set(list('abcde'))
        labels = ([np.asarray(list(labelset)) for _ in range(5)])

        train_dur = None
        val_dur = None
        test_dur = None

        with self.assertRaises(ValueError):
            vak.io.split.train_test_dur_split_inds(durs,
                                                   labels,
                                                   labelset,
                                                   train_dur,
                                                   test_dur,
                                                   val_dur)

    def test_train_test_dur_split_only_val_raises(self):
        durs = (5, 5, 5, 5, 5)
        labelset = set(list('abcde'))
        labels = ([np.asarray(list(labelset)) for _ in range(5)])

        train_dur = None
        val_dur = 100
        test_dur = None

        # because we only specified duration for validation set
        with self.assertRaises(OnlyValDurError):
            vak.io.split.train_test_dur_split_inds(durs,
                                                   labels,
                                                   labelset,
                                                   train_dur,
                                                   test_dur,
                                                   val_dur)

    def test_train_test_dur_split_negative_dur_raises(self):
        durs = (5, 5, 5, 5, 5)
        labelset = set(list('abcde'))
        labels = ([np.asarray(list(labelset)) for _ in range(5)])

        train_dur = -2
        test_dur = None
        val_dur = 100

        # because negative duration is invalid
        with self.assertRaises(InvalidDurationError):
            vak.io.split.train_test_dur_split_inds(durs,
                                                   labels,
                                                   labelset,
                                                   train_dur,
                                                   test_dur,
                                                   val_dur)

    def test_train_test_dur_split_specd_dur_gt_raises(self):
        durs = (5, 5, 5, 5, 5)
        labelset = set(list('abcde'))
        labels = ([np.asarray(list(labelset)) for _ in range(5)])

        train_dur = 100
        test_dur = 100
        val_dur = 100
        # because total splits duration is greater than dataset duration
        with self.assertRaises(SplitsDurationGreaterThanDatasetDurationError):
            vak.io.split.train_test_dur_split_inds(durs,
                                                   labels,
                                                   labelset,
                                                   train_dur,
                                                   test_dur,
                                                   val_dur)

    def test_train_test_dur_split_mat_train_test(self):
        vds = vak.io.dataframe.from_files(spect_format='mat',
                                          spect_dir=spect_dir_mat,
                                          annot_list=annot_list_mat,
                                          load_spects=False)

        train_dur = 200
        test_dur = 200

        train_vds, test_vds = vak.io.split.train_test_dur_split(vds,
                                                                labelset=self.labelset_mat,
                                                                train_dur=train_dur,
                                                                test_dur=test_dur)
        for vds_out in (train_vds, test_vds):
            self.assertTrue(type(vds_out) == Dataset)

        train_dur_out = sum([voc.duration for voc in train_vds.voc_list])
        self.assertTrue(train_dur_out >= train_dur)
        test_dur_out = sum([voc.duration for voc in test_vds.voc_list])
        self.assertTrue(test_dur_out >= test_dur)

    def test_train_test_dur_split_mat_train(self):
        vds = vak.io.dataframe.from_files(spect_format='mat',
                                          spect_dir=spect_dir_mat,
                                          annot_list=annot_list_mat,
                                          load_spects=False)

        train_dur = 200

        train_vds, test_vds = vak.io.split.train_test_dur_split(vds,
                                                                labelset=self.labelset_mat,
                                                                train_dur=train_dur)
        for vds_out in (train_vds, test_vds):
            self.assertTrue(type(vds_out) == Dataset)

        train_dur_out = sum([voc.duration for voc in train_vds.voc_list])
        test_dur_out = sum([voc.duration for voc in test_vds.voc_list])
        vds_dur = sum([voc.duration for voc in vds.voc_list])

        self.assertTrue(train_dur_out >= train_dur)
        self.assertTrue(
            isclose(test_dur_out, vds_dur - train_dur_out)
        )

    def test_train_test_dur_split_mat_test(self):
        vds = vak.io.dataframe.from_files(spect_format='mat',
                                          spect_dir=spect_dir_mat,
                                          annot_list=annot_list_mat,
                                          load_spects=False)

        test_dur = 200

        train_vds, test_vds = vak.io.split.train_test_dur_split(vds,
                                                                labelset=self.labelset_mat,
                                                                test_dur=test_dur)
        for vds_out in (train_vds, test_vds):
            self.assertTrue(type(vds_out) == Dataset)

        train_dur_out = sum([voc.duration for voc in train_vds.voc_list])
        test_dur_out = sum([voc.duration for voc in test_vds.voc_list])
        vds_dur = sum([voc.duration for voc in vds.voc_list])

        self.assertTrue(
            isclose(train_dur_out, vds_dur - test_dur_out)
        )
        self.assertTrue(test_dur_out >= test_dur)


if __name__ == '__main__':
    unittest.main()
