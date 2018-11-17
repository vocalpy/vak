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


class TestLearncurve(unittest.TestCase):
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
        # make sure learncurve runs without crashing.
        config = songdeck.config.parse.parse_config(self.tmp_config_path)
        songdeck.cli.learncurve(train_data_dict_path=self.train_data_dict_path,
                                val_data_dict_path=self.val_data_dict_path,
                                spect_params=config.spect_params,
                                total_train_set_duration=config.data.total_train_set_dur,
                                train_set_durs=config.train.train_set_durs,
                                num_replicates=config.train.num_replicates,
                                num_epochs=config.train.num_epochs,
                                config_file=self.tmp_config_path,
                                networks=config.networks,
                                val_error_step=config.train.val_error_step,
                                checkpoint_step=config.train.checkpoint_step,
                                patience=config.train.patience,
                                save_only_single_checkpoint_file=config.train.save_only_single_checkpoint_file,
                                normalize_spectrograms=config.train.normalize_spectrograms,
                                use_train_subsets_from_previous_run=config.train.use_train_subsets_from_previous_run,
                                previous_run_path=config.train.previous_run_path,
                                root_results_dir=config.output.root_results_dir)


if __name__ == '__main__':
    unittest.main()
