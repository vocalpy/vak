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

    def test_prep_func(self):
        # all this does is make sure the functions runs without crashing.
        # need to write tests for functions that prep calls as well
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                   transform_type='log_spect')
        vak.cli.prep(labelset=list('iabcdefghjk'),
                     all_labels_are_int=False,
                     data_dir=data_dir,
                     total_train_set_dur=20,
                     val_dur=10,
                     test_dur=20,
                     config_file=self.tmp_config_path,
                     silent_gap_label=0,
                     skip_files_with_labels_not_in_labelset=True,
                     output_dir=self.tmp_output_dir,
                     mat_spect_files_path=None,
                     mat_spects_annotation_file=None,
                     spect_params=spect_params)
        data_dicts = glob(os.path.join(self.tmp_output_dir, 'spectrograms*', '*dict*'))
        assert len(data_dicts) == 3


if __name__ == '__main__':
    unittest.main()
