from glob import glob
import os
import unittest
from math import isclose

import crowsetta
from scipy.io import loadmat

from vak.evfuncs import load_cbin
from vak.utils.splitalgos import brute_force
from vak.io.annotation import files_from_dir
from vak.utils.general import timebin_dur_from_vec

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
scribe_cbin = crowsetta.Transcriber(annot_format='notmat')
annot_list_cbin = scribe_cbin.from_file(annot_file=annot_files_cbin)
labelset_cbin = set(list('iabcdefghjk'))
durs_cbin = []
labels_cbin = []
for audio_file, annot in zip(audio_files_cbin, annot_list_cbin):
    if set(annot.seq.labels).issubset(labelset_cbin):
        labels_cbin.append(annot.seq.labels)
        fs, data = load_cbin(audio_file)
        durs_cbin.append(data.shape[0] / fs)

spect_dir_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
spect_files_mat = glob(os.path.join(spect_dir_mat, '*.mat'))
annot_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')
scribe_yarden = crowsetta.Transcriber(annot_format='yarden')
annot_list_mat = scribe_yarden.from_file(annot_mat)
labelset_mat = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}
durs_mat = []
labels_mat = []
for spect_file_mat, annot in zip(spect_files_mat, annot_list_mat):
    if set(annot.seq.labels).issubset(labelset_mat):
        labels_mat.append(annot.seq.labels)
        mat_dict = loadmat(spect_file_mat)
        timebin_dur = timebin_dur_from_vec(mat_dict['t'])
        dur = mat_dict['s'].shape[-1] * timebin_dur
        durs_mat.append(dur)


class TestBruteforce(unittest.TestCase):
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

    def test_bruteforce_train_test_val_mock(self):
        train_dur = 2
        test_dur = 2
        val_dur = 1
        durs = (1, 1, 1, 1, 1)
        labelset = set(list('abcde'))
        labels = [list('abcde') for _ in range(5)]

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = brute_force(durs, labels, labelset, train_dur, val_dur, test_dur)

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

    def test_bruteforce_train_test_val_cbin(self):
        train_dur = 35
        val_dur = 20
        test_dur = 35

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = brute_force(self.durs_cbin,
                                                          self.labels_cbin,
                                                          self.labelset_cbin,
                                                          train_dur,
                                                          val_dur,
                                                          test_dur)

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

    def test_bruteforce_train_test_val_mat(self):
        train_dur = 200
        val_dur = 100
        test_dur = 200

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = brute_force(self.durs_mat,
                                                          self.labels_mat,
                                                          self.labelset_mat,
                                                          train_dur,
                                                          val_dur,
                                                          test_dur)

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

    def test_bruteforce_train_mock(self):
        train_dur = 2
        val_dur = None
        test_dur = -1
        durs = (1, 1, 1, 1, 1)
        labelset = set(list('abcde'))
        labels = [list('abcde') for _ in range(5)]

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = brute_force(durs, labels, labelset, train_dur, val_dur, test_dur)

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

    def test_bruteforce_train_cbin(self):
        train_dur = 35
        val_dur = None
        test_dur = -1

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = brute_force(self.durs_cbin,
                                                          self.labels_cbin,
                                                          self.labelset_cbin,
                                                          train_dur,
                                                          val_dur,
                                                          test_dur)

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

    def test_bruteforce_train_mat(self):
        train_dur = 300
        val_dur = None
        test_dur = -1

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = brute_force(self.durs_mat,
                                                          self.labels_mat,
                                                          self.labelset_mat,
                                                          train_dur,
                                                          val_dur,
                                                          test_dur)

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

    def test_bruteforce_test_mock(self):
        train_dur = -1
        test_dur = 2
        val_dur = None
        durs = (1, 1, 1, 1, 1)
        labelset = set(list('abcde'))
        labels = [list('abcde') for _ in range(5)]

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = brute_force(durs, labels, labelset, train_dur, val_dur, test_dur)

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

    def test_bruteforce_test_cbin(self):
        train_dur = -1
        val_dur = None
        test_dur = 25

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = brute_force(self.durs_cbin,
                                                          self.labels_cbin,
                                                          self.labelset_cbin,
                                                          train_dur,
                                                          val_dur,
                                                          test_dur)

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

    def test_bruteforce_test_mat(self):
        train_dur = -1
        val_dur = None
        test_dur = 200

        for _ in range(NUM_SAMPLES):
            train_inds, val_inds, test_inds = brute_force(self.durs_mat,
                                                          self.labels_mat,
                                                          self.labelset_mat,
                                                          train_dur,
                                                          val_dur,
                                                          test_dur)

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


if __name__ == '__main__':
    unittest.main()
