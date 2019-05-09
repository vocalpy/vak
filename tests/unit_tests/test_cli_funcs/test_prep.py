"""tests for vak.cli.make_data module"""
import os
import tempfile
import shutil
from glob import glob
import unittest

import joblib


import vak.cli.prep
from vak.config.spectrogram import SpectConfig

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')


class TestMakeData(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()
        a_config = os.path.join(TEST_DATA_DIR, 'configs', 'test_learncurve_config.ini')
        self.tmp_config_path = os.path.join(TEST_DATA_DIR, 'configs', 'tmp_config.ini')
        shutil.copy(a_config, self.tmp_config_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        os.remove(self.tmp_config_path)

    def test_prep_with_audio_cbin(self):
        # all this does is make sure the functions runs without crashing.
        # need to write tests for functions that prep calls as well
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                   transform_type='log_spect')
        labelset = list('iabcdefghjk')

        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     total_train_set_dur=20,
                     val_dur=10,
                     test_dur=20,
                     config_file=self.tmp_config_path,
                     annot_format='notmat',
                     silent_gap_label=0,
                     skip_files_with_labels_not_in_labelset=True,
                     output_dir=self.tmp_output_dir,
                     audio_format='cbin',
                     array_format=None,
                     annot_file=None,
                     spect_params=spect_params)

        data_dicts = glob(os.path.join(self.tmp_output_dir, 'spectrograms*', '*dict*'))
        assert len(data_dicts) == 3

    def test_prep_with_array_mat(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb11', 'spect')
        annot_file = os.path.join(TEST_DATA_DIR, 'mat', 'llb11', 'llb11_annot_subset.mat')
        labelset = {1, 4, 5, 9, 10, 13, 15, 16, 17, 19, 20, 21, 23, 27}
        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     total_train_set_dur=20,
                     val_dur=10,
                     test_dur=20,
                     config_file=self.tmp_config_path,
                     annot_format='yarden',
                     silent_gap_label=0,
                     skip_files_with_labels_not_in_labelset=True,
                     output_dir=self.tmp_output_dir,
                     audio_format=None,
                     array_format='mat',
                     annot_file=annot_file,
                     spect_params=None)


if __name__ == '__main__':
    unittest.main()
