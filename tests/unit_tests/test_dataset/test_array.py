import os
import unittest
from glob import glob

import numpy as np
import crowsetta

import vak.dataset.spect
from vak.dataset.classes import VocalizationDataset, Vocalization, Spectrogram


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestArray(unittest.TestCase):
    def setUp(self):
        self.array_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        self.array_files = glob(os.path.join(self.array_dir, '*.mat'))
        self.array_format = 'mat'

        self.annot_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')
        self.scribe = crowsetta.Transcriber(voc_format='yarden')
        self.annot_list = self.scribe.to_seq(self.annot_mat)
        self.labelset_mat = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

    def _check_vocal_dataset_returned_by_from_arr_files(self, vocal_dataset, load_spects=True):
        """assertions that are shared across unit tests for vak.dataset.array.from_arr_files"""
        self.assertTrue(
            type(vocal_dataset) == VocalizationDataset
        )

        self.assertTrue(
            all([type(voc) == Vocalization for voc in vocal_dataset.voc_list])
        )

        self.assertTrue(
            all([hasattr(voc, 'spect') for voc in vocal_dataset.voc_list])
        )

        array_file_basenames = [os.path.basename(arr_path) for arr_path in self.array_files]
        spect_files = [os.path.basename(voc.spect_file)
                       for voc in vocal_dataset.voc_list]
        self.assertTrue(
            all([spect_file in array_file_basenames for spect_file in spect_files])
        )

        if load_spects:
            self.assertTrue(
                all([type(voc.spect.array) == np.ndarray for voc in vocal_dataset.voc_list])
            )
            self.assertTrue(
                all([type(voc.spect) == Spectrogram for voc in vocal_dataset.voc_list])
            )

        elif load_spects is False:
            self.assertTrue(
                all([voc.spect is None for voc in vocal_dataset.voc_list])
            )

        # if all assertTrues were True
        return True

    def test_array_dir_annot(self):
        vocal_dataset = vak.dataset.spect.from_files(self.array_format,
                                                     spect_dir=self.array_dir,
                                                     annot_list=self.annot_list,
                                                     load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_arr_files(vocal_dataset)
        )

        # make sure we're not loading arrays when we don't want to
        vocal_dataset = vak.dataset.spect.from_files(self.array_format,
                                                     spect_dir=self.array_dir,
                                                     annot_list=self.annot_list,
                                                     load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_arr_files(vocal_dataset,
                                                                 load_spects=False)
        )

    def test_array_files_annot(self):
        vocal_dataset = vak.dataset.spect.from_files(self.array_format,
                                                     spect_files=self.array_files,
                                                     annot_list=self.annot_list,
                                                     load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_arr_files(vocal_dataset)
        )

        # make sure we're not loading arrays when we don't want to
        vocal_dataset = vak.dataset.spect.from_files(self.array_format,
                                                     spect_files=self.array_files,
                                                     annot_list=self.annot_list,
                                                     load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_arr_files(vocal_dataset,
                                                                 load_spects=False)
        )

    def test_array_annot_map(self):
        array_annot_map = dict(zip(self.array_files, self.annot_list))
        vocal_dataset = vak.dataset.spect.from_files(self.array_format,
                                                     array_annot_map=array_annot_map,
                                                     load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_arr_files(vocal_dataset)
        )

        # make sure we're not loading arrays when we don't want to
        vocal_dataset = vak.dataset.spect.from_files(self.array_format,
                                                     array_annot_map=array_annot_map,
                                                     load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_arr_files(vocal_dataset,
                                                                 load_spects=False)
        )

    def test_bad_inputs_raise(self):
        # invalid array format
        with self.assertRaises(ValueError):
            vak.dataset.spect.from_files(spect_format='npy',
                                         spect_dir=self.array_dir,
                                         spect_files=self.array_files,
                                         annot_list=self.annot_list,
                                         load_spects=True)

        # can't specify both dir and list
        with self.assertRaises(ValueError):
            vak.dataset.spect.from_files(self.array_format,
                                         spect_dir=self.array_dir,
                                         spect_files=self.array_files,
                                         annot_list=self.annot_list,
                                         load_spects=True)

        # can't specify both dir and array_annot_map
        array_annot_map = dict(zip(self.array_files, self.annot_list))
        with self.assertRaises(ValueError):
            vak.dataset.spect.from_files(self.array_format,
                                         spect_dir=self.array_dir,
                                         array_annot_map=array_annot_map,
                                         load_spects=True)

        # can't specify both list and array_annot_map
        with self.assertRaises(ValueError):
            vak.dataset.spect.from_files(self.array_format,
                                         spect_files=self.array_files,
                                         array_annot_map=array_annot_map,
                                         load_spects=True)

        # can't specify both annotations list and array_annot_map
        with self.assertRaises(ValueError):
            vak.dataset.spect.from_files(self.array_format,
                                         array_annot_map=array_annot_map,
                                         annot_list=self.annot_list,
                                         load_spects=True)


if __name__ == '__main__':
    unittest.main()
