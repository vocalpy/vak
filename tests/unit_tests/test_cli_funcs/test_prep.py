"""tests for vak.cli.prep module"""
import os
import tempfile
import shutil
from glob import glob
import unittest

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
                     total_train_set_dur=35,
                     val_dur=20,
                     test_dur=35,
                     config_file=self.tmp_config_path,
                     annot_format='notmat',
                     output_dir=self.tmp_output_dir,
                     audio_format='cbin',
                     spect_format=None,
                     annot_file=None,
                     spect_params=spect_params)

        vds_paths = glob(os.path.join(self.tmp_output_dir, '*vds.json'))
        self.assertTrue(len(vds_paths) == 4)

    def test_prep_with_array_mat(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        annot_file = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')
        labelset = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}
        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     total_train_set_dur=200,
                     val_dur=100,
                     test_dur=200,
                     config_file=self.tmp_config_path,
                     annot_format='yarden',
                     output_dir=self.tmp_output_dir,
                     audio_format=None,
                     spect_format='mat',
                     annot_file=annot_file,
                     spect_params=None)

        vds_paths = glob(os.path.join(self.tmp_output_dir, '*vds.json'))
        self.assertTrue(len(vds_paths) == 4)


if __name__ == '__main__':
    unittest.main()
