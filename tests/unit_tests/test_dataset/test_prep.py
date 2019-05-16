"""tests for vak.dataset.prep module"""
import os
import tempfile
import shutil
import unittest

import vak.dataset.prep
from vak.config.spectrogram import SpectConfig
from vak.dataset.classes import VocalizationDataset

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')


class TestPrep(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)

    def test_prep_with_audio_cbin(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                   transform_type='log_spect')
        labelset = list('iabcdefghjk')
        vds_fname = 'test.json'

        vds, vds_path = vak.dataset.prep(labelset=labelset,
                                         data_dir=data_dir,
                                         annot_format='notmat',
                                         skip_files_with_labels_not_in_labelset=True,
                                         output_dir=self.tmp_output_dir,
                                         save_vds=True,
                                         vds_fname=vds_fname,
                                         return_vds=True,
                                         return_path=True,
                                         load_spects=False,
                                         audio_format='cbin',
                                         spect_format=None,
                                         annot_file=None,
                                         spect_params=spect_params)

        self.assertTrue(type(vds) == VocalizationDataset)
        json_fname = os.path.join(self.tmp_output_dir, vds_fname)
        self.assertTrue(
            os.path.isfile(json_fname)
        )
        vds_loaded = VocalizationDataset.load(json_fname=json_fname)
        for voc, voc_loaded in zip(vds.voc_list, vds_loaded.voc_list):
            self.assertTrue(
                voc.audio_file == voc_loaded.audio_file
            )
            self.assertTrue(
                voc.audio_file == voc_loaded.audio_file
            )

    def test_prep_with_array_mat(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        annot_file = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')
        labelset = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}
        vds_fname = 'test.json'
        vds, vds_path = vak.dataset.prep(labelset=labelset,
                                         data_dir=data_dir,
                                         annot_format='yarden',
                                         skip_files_with_labels_not_in_labelset=True,
                                         output_dir=self.tmp_output_dir,
                                         save_vds=True,
                                         vds_fname=vds_fname,
                                         return_vds=True,
                                         return_path=True,
                                         load_spects=False,
                                         audio_format=None,
                                         spect_format='mat',
                                         annot_file=annot_file,
                                         spect_params=None)

        self.assertTrue(type(vds) == VocalizationDataset)
        json_fname = os.path.join(self.tmp_output_dir, vds_fname)
        self.assertTrue(
            os.path.abspath(vds_path) == os.path.abspath(json_fname)
        )
        self.assertTrue(
            os.path.isfile(json_fname)
        )
        vds_loaded = VocalizationDataset.load(json_fname=json_fname)
        for voc, voc_loaded in zip(vds.voc_list, vds_loaded.voc_list):
            self.assertTrue(
                voc.audio_file == voc_loaded.audio_file
            )
            self.assertTrue(
                voc.spect_file == voc_loaded.spect_file
            )


if __name__ == '__main__':
    unittest.main()
