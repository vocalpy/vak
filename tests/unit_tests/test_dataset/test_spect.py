import os
from pathlib import Path
import unittest

import numpy as np
import crowsetta

import vak.dataset.dataframe
from vak.dataset.classes import Dataset, Vocalization, MetaSpect
from vak.config.validators import VALID_AUDIO_FORMATS

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestFindAudioFname(unittest.TestCase):
    """class to test find_audio_fname function"""
    def setUp(self):
        # ---- in .mat files -------------------------------
        self.spect_dir_mat = TEST_DATA_DIR.joinpath('mat', 'llb3', 'spect')
        self.spect_list_mat = list(self.spect_dir_mat.glob('*.mat'))
        self.spect_list_mat = [str(path) for path in self.spect_list_mat]

        # ---- in .npz files, made from .cbin audio files -------------------------------
        self.spect_dir_npz = list(TEST_DATA_DIR.joinpath('vds').glob(
            'spectrograms_generated*')
        )
        self.spect_dir_npz = self.spect_dir_npz[0]
        self.spect_list_npz = list(self.spect_dir_npz.glob('*.spect.npz'))
        self.spect_list_npz = [str(path) for path in self.spect_list_npz]

    def test_with_mat(self):
        audio_fnames = [vak.dataset.dataframe.find_audio_fname(spect_path)
                        for spect_path in self.spect_list_mat]
        for mat_spect_path, audio_fname in zip(self.spect_list_mat, audio_fnames):
            # make sure we gout out a filename that was actually in spect_path
            self.assertTrue(audio_fname in mat_spect_path)
            # make sure it's some valid audio format
            self.assertTrue(
                Path(audio_fname).suffix.replace('.', '') in VALID_AUDIO_FORMATS
            )

    def test_with_npz(self):
        audio_fnames = [vak.dataset.dataframe.find_audio_fname(spect_path)
                        for spect_path in self.spect_list_npz]
        for npz_spect_path, audio_fname in zip(self.spect_list_npz, audio_fnames):
            # make sure we gout out a filename that was actually in spect_path
            self.assertTrue(audio_fname in npz_spect_path)
            self.assertTrue(
                Path(audio_fname).suffix.replace('.', '') in VALID_AUDIO_FORMATS
            )


class TestFromFiles(unittest.TestCase):
    """class to tests spect.from_files function"""
    def setUp(self):
        self.spect_dir = TEST_DATA_DIR.joinpath('mat', 'llb3', 'spect')
        self.spect_files = self.spect_dir.glob('*.mat')
        self.spect_files = sorted([str(path) for path in self.spect_files])
        self.spect_format = 'mat'

        self.annot_mat = TEST_DATA_DIR.joinpath('mat', 'llb3', 'llb3_annot_subset.mat')
        self.annot_mat = str(self.annot_mat)
        self.scribe = crowsetta.Transcriber(voc_format='yarden')
        self.annot_list = self.scribe.to_seq(self.annot_mat)
        self.labelset_mat = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

    def _check_vocal_dataset_returned_by_from_spect_files(self, vocal_dataset, load_spects=True):
        """assertions that are shared across unit tests for vak.dataset.spect.from_files"""
        self.assertTrue(
            type(vocal_dataset) == Dataset
        )

        self.assertTrue(
            all([type(voc) == Vocalization for voc in vocal_dataset.voc_list])
        )

        self.assertTrue(
            all([hasattr(voc, 'metaspect') for voc in vocal_dataset.voc_list])
        )

        spect_files_from_test_data = [os.path.basename(spect_path)
                                      for spect_path in self.spect_files]
        spect_files_from_vds = [os.path.basename(voc.spect_path)
                                for voc in vocal_dataset.voc_list]

        self.assertTrue(
            all([spect_file in spect_files_from_test_data
                 for spect_file in spect_files_from_vds])
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
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         spect_dir=self.spect_dir,
                                                         labelset=self.labelset_mat,
                                                         annot_list=self.annot_list,
                                                         load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset)
        )

        # make sure we're not loading spects when we don't want to
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         spect_dir=self.spect_dir,
                                                         labelset=self.labelset_mat,
                                                         annot_list=self.annot_list,
                                                         load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset,
                                                                   load_spects=False)
        )

    def test_spect_dir_annot_no_labelset(self):
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         spect_dir=self.spect_dir,
                                                         labelset=None,
                                                         annot_list=self.annot_list,
                                                         load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset)
        )

        # make sure we're not loading spects when we don't want to
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         spect_dir=self.spect_dir,
                                                         labelset=None,
                                                         annot_list=self.annot_list,
                                                         load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset,
                                                                   load_spects=False)
        )

    def test_spect_dir_without_annot(self):
        # make sure we can make a dataset from spectrogram files without annotations,
        # e.g. if we're going to predict the annotations using the spectrograms
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         spect_dir=self.spect_dir,
                                                         annot_list=None,
                                                         load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset)
        )

    def test_spect_files_annot(self):
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         spect_files=self.spect_files,
                                                         labelset=self.labelset_mat,
                                                         annot_list=self.annot_list,
                                                         load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset)
        )

        # make sure we're not loading spects when we don't want to
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         spect_files=self.spect_files,
                                                         labelset=self.labelset_mat,
                                                         annot_list=self.annot_list,
                                                         load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset,
                                                                   load_spects=False)
        )

    def test_spect_files_annot_no_labelset(self):
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         spect_files=self.spect_files,
                                                         labelset=None,
                                                         annot_list=self.annot_list,
                                                         load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset)
        )

        # make sure we're not loading spects when we don't want to
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         spect_files=self.spect_files,
                                                         labelset=None,
                                                         annot_list=self.annot_list,
                                                         load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset,
                                                                   load_spects=False)
        )

    def test_spect_annot_map(self):
        spect_annot_map = dict(zip(self.spect_files, self.annot_list))
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         labelset=self.labelset_mat,
                                                         spect_annot_map=spect_annot_map,
                                                         load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset)
        )

        # make sure we're not loading spects when we don't want to
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         labelset=self.labelset_mat,
                                                         spect_annot_map=spect_annot_map,
                                                         load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset,
                                                                   load_spects=False)
        )

    def test_spect_annot_map_no_labelset(self):
        spect_annot_map = dict(zip(self.spect_files, self.annot_list))
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         labelset=None,
                                                         spect_annot_map=spect_annot_map,
                                                         load_spects=True)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset)
        )

        # make sure we're not loading spects when we don't want to
        vocal_dataset = vak.dataset.dataframe.from_files(self.spect_format,
                                                         labelset=None,
                                                         spect_annot_map=spect_annot_map,
                                                         load_spects=False)
        self.assertTrue(
            self._check_vocal_dataset_returned_by_from_spect_files(vocal_dataset,
                                                                   load_spects=False)
        )

    def test_bad_inputs_raise(self):
        # must specify one of: spect dir, spect files, or spect files/annotations mapping
        with self.assertRaises(ValueError):
            vak.dataset.dataframe.from_files(spect_format='npz',
                                             spect_dir=None,
                                             spect_files=None,
                                             annot_list=self.annot_list,
                                             spect_annot_map=None,
                                             load_spects=True)
        # invalid spect format
        with self.assertRaises(ValueError):
            vak.dataset.dataframe.from_files(spect_format='npy',
                                             spect_dir=self.spect_dir,
                                             spect_files=self.spect_files,
                                             annot_list=self.annot_list,
                                             load_spects=True)

        # can't specify both dir and list
        with self.assertRaises(ValueError):
            vak.dataset.dataframe.from_files(self.spect_format,
                                             spect_dir=self.spect_dir,
                                             spect_files=self.spect_files,
                                             annot_list=self.annot_list,
                                             load_spects=True)

        # can't specify both dir and spect_annot_map
        spect_annot_map = dict(zip(self.spect_files, self.annot_list))
        with self.assertRaises(ValueError):
            vak.dataset.dataframe.from_files(self.spect_format,
                                             spect_dir=self.spect_dir,
                                             spect_annot_map=spect_annot_map,
                                             load_spects=True)

        # can't specify both list and spect_annot_map
        with self.assertRaises(ValueError):
            vak.dataset.dataframe.from_files(self.spect_format,
                                             spect_files=self.spect_files,
                                             spect_annot_map=spect_annot_map,
                                             load_spects=True)

        # can't specify both annotations list and spect_annot_map
        with self.assertRaises(ValueError):
            vak.dataset.dataframe.from_files(self.spect_format,
                                             spect_annot_map=spect_annot_map,
                                             annot_list=self.annot_list,
                                             load_spects=True)



if __name__ == '__main__':
    unittest.main()
