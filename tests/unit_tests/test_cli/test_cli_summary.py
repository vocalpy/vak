"""tests for songdeck.cli.learncurve module"""
import os
import tempfile
import shutil
from glob import glob
import unittest
from configparser import ConfigParser

import joblib

import songdeck.cli.make_data
from songdeck.config.spectrogram import SpectConfig

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestSummary(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()
        # Makefile copies Makefile_config to a tmp version (that gets changed by make_data
        # and other functions)
        tmp_makefile_config = os.path.join(SETUP_SCRIPTS_DIR, 'tmp_Makefile_config.ini')
        # Now we want a copy (of the changed version) to use for tests
        # since this is what the test data was made with
        self.tmp_config_path = os.path.join(TEST_DATA_DIR, 'configs', 'tmp_config.ini')
        shutil.copy(tmp_makefile_config, self.tmp_config_path)
        test_data_spects_path = glob(os.path.join(TEST_DATA_DIR,
                                                  'spects',
                                                  'spectrograms_*'))
        self.assertTrue(len(test_data_spects_path) == 1)
        test_data_spects_path = test_data_spects_path[0]
        self.train_data_dict_path = os.path.join(test_data_spects_path,'train_data_dict')
        self.assertTrue(os.path.isfile(self.train_data_dict_path))
        self.val_data_dict_path = os.path.join(test_data_spects_path,'val_data_dict')
        self.assertTrue(os.path.isfile(self.val_data_dict_path))


    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        os.remove(self.tmp_config_path)

    def test_learncurve_func(self):
        # make sure cli.summary runs without crashing.
        config = songdeck.config.parse.parse_config(self.tmp_config_path)
        songdeck.cli.summary(results_dirname=config.output.results_dir_made_by_main_script,
                             networks=config.networks,
                             train_set_durs=config.train.train_set_durs,
                             num_replicates=config.train.num_replicates,
                             labelset=config.data.labelset,
                             test_data_dict_path=config.train.test_data_dict_path,
                             normalize_spectrograms=config.train.normalize_spectrograms)


if __name__ == '__main__':
    unittest.main()
