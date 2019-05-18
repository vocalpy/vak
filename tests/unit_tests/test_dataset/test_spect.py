import os
import unittest
from glob import glob

import numpy as np
import crowsetta

import vak.dataset.spect
from vak.dataset.classes import VocalizationDataset, Vocalization, MetaSpect


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestSpect(unittest.TestCase):
    def setUp(self):
        self.spect_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        self.spect_files = glob(os.path.join(self.spect_dir, '*.mat'))
        self.spect_format = 'mat'

        self.annot_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')
        self.scribe = crowsetta.Transcriber(voc_format='yarden')
        self.annot_list = self.scribe.to_seq(self.annot_mat)
        self.labelset_mat = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

    def _check_vocal_dataset_returned_by_from_spect_files(self, vocal_dataset, load_spects=True):
        """assertions that are shared across unit tests for vak.dataset.spect.from_files"""
        self.assertTrue(
            type(vocal_dataset) == VocalizationDataset
        )

        self.assertTrue(
            all([type(voc) == Vocalization for voc in vocal_dataset.voc_list])
        )

        self.assertTrue(
            all([hasattr(voc, 'metaspect') for voc in vocal_dataset.voc_list])
        )

        spect_files_from_test_data = [os.path.basename(spect_path) for spect_path in self.spect_files]
        spect_files_from_vds = [os.path.basename(voc.spect_path)
                                for voc in vocal_dataset.voc_list]
        self.assertTrue(
            all([spect_file in spect_files_from_test_data for spect_file in spect_files_from_vds])
        )

        if load_spects:
            self.assertTrue(
                all([type(voc.metaspect.spect) == np.ndarray for voc in vocal_dataset.voc_list])
            )
            self.assertTrue(
                all([type(voc.metaspect) == MetaSpect for voc in vocal_dataset.voc_list])
            )

        elif load_spects is False:
            self.assertTrue(
                all([voc.metaspect is None for voc in vocal_dataset.voc_list])
            )

        # if all assertTrues were True
        return True

    def test_spect_dir_annot(self):
        vocal_dataset = vak.dataset.spect.from_files(self.spect_format,
                                                     spect_dir=self.spect_dir,
                                                     annot_list=self.annot_list,
                                                     load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset)
        )

        # make sure we're not loading spects when we don't want to
        vocal_dataset = vak.dataset.spect.from_files(self.spect_format,
                                                     spect_dir=self.spect_dir,
                                                     annot_list=self.annot_list,
                                                     load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset,
                                                                   load_spects=False)
        )

    def test_spect_files_annot(self):
        vocal_dataset = vak.dataset.spect.from_files(self.spect_format,
                                                     spect_files=self.spect_files,
                                                     annot_list=self.annot_list,
                                                     load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset)
        )

        # make sure we're not loading spects when we don't want to
        vocal_dataset = vak.dataset.spect.from_files(self.spect_format,
                                                     spect_files=self.spect_files,
                                                     annot_list=self.annot_list,
                                                     load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset,
                                                                   load_spects=False)
        )

    def test_spect_annot_map(self):
        spect_annot_map = dict(zip(self.spect_files, self.annot_list))
        vocal_dataset = vak.dataset.spect.from_files(self.spect_format,
                                                     spect_annot_map=spect_annot_map,
                                                     load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset)
        )

        # make sure we're not loading spects when we don't want to
        vocal_dataset = vak.dataset.spect.from_files(self.spect_format,
                                                     spect_annot_map=spect_annot_map,
                                                     load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset,
                                                                   load_spects=False)
        )

    def test_bad_inputs_raise(self):
        # invalid spect format
        with self.assertRaises(ValueError):
            vak.dataset.spect.from_files(spect_format='npy',
                                         spect_dir=self.spect_dir,
                                         spect_files=self.spect_files,
                                         annot_list=self.annot_list,
                                         load_spects=True)

        # can't specify both dir and list
        with self.assertRaises(ValueError):
            vak.dataset.spect.from_files(self.spect_format,
                                         spect_dir=self.spect_dir,
                                         spect_files=self.spect_files,
                                         annot_list=self.annot_list,
                                         load_spects=True)

        # can't specify both dir and spect_annot_map
        spect_annot_map = dict(zip(self.spect_files, self.annot_list))
        with self.assertRaises(ValueError):
            vak.dataset.spect.from_files(self.spect_format,
                                         spect_dir=self.spect_dir,
                                         spect_annot_map=spect_annot_map,
                                         load_spects=True)

        # can't specify both list and spect_annot_map
        with self.assertRaises(ValueError):
            vak.dataset.spect.from_files(self.spect_format,
                                         spect_files=self.spect_files,
                                         spect_annot_map=spect_annot_map,
                                         load_spects=True)

        # can't specify both annotations list and spect_annot_map
        with self.assertRaises(ValueError):
            vak.dataset.spect.from_files(self.spect_format,
                                         spect_annot_map=spect_annot_map,
                                         annot_list=self.annot_list,
                                         load_spects=True)


if __name__ == '__main__':
    unittest.main()
