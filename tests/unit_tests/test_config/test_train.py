"""tests for vak.config.train module"""
import unittest
import os
import tempfile
from configparser import ConfigParser
import shutil

import vak.config.train
import vak.utils


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')
TEST_CONFIGS_DIR = os.path.join(TEST_DATA_DIR, 'configs')


class TestParseTrainConfig(unittest.TestCase):
    def setUp(self):
        _, self.tmp_train_data_path = tempfile.mkstemp()
        _, self.tmp_val_data_path = tempfile.mkstemp()
        _, self.tmp_test_data_path = tempfile.mkstemp()

        self.config_file = os.path.join(TEST_DATA_DIR, 'configs', 'test_learncurve_config.ini')
        self.config_obj = ConfigParser()
        self.config_obj.read(self.config_file)
        self.config_obj['TRAIN']['train_data_path'] = self.tmp_train_data_path
        self.config_obj['TRAIN']['val_data_path'] = self.tmp_val_data_path
        self.config_obj['TRAIN']['test_data_path'] = self.tmp_test_data_path

    def tearDown(self):
        os.remove(self.tmp_train_data_path)
        os.remove(self.tmp_val_data_path)
        os.remove(self.tmp_test_data_path)

    def test_parse_train_config_returns_TrainConfig_instance(self):
        predict_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(type(predict_config_obj) == vak.config.train.TrainConfig)

    def test_no_networks_raises(self):
        self.config_obj.remove_option('TRAIN', 'networks')
        with self.assertRaises(KeyError):
            vak.config.train.parse_train_config(self.config_obj, self.config_file)

    def test_network_not_installed_raises(self):
        self.config_obj['TRAIN']['networks'] = 'NotInstalledNet, OtherNotInstalledNet'
        with self.assertRaises(TypeError):
            vak.config.train.parse_train_config(self.config_obj, self.config_file)

    def test_no_train_path_raises(self):
        self.config_obj.remove_option('TRAIN', 'train_data_path')
        with self.assertRaises(KeyError):
            vak.config.train.parse_train_config(self.config_obj, self.config_file)

    def test_train_set_durs_default(self):
        self.config_obj.remove_option('TRAIN', 'train_set_durs')
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.train_set_durs is None)

    def test_replicates_default(self):
        self.config_obj.remove_option('TRAIN', 'replicates')
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.num_replicates is None)

    def test_val_data_dict_path_default(self):
        self.config_obj.remove_option('TRAIN', 'val_data_path')
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.val_data_dict_path is None)

    def test_test_data_dict_path_default(self):
        self.config_obj.remove_option('TRAIN', 'test_data_path')
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.test_data_dict_path is None)

    def test_val_error_step_default(self):
        self.config_obj.remove_option('TRAIN', 'val_error_step')
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.val_error_step is None)

    def test_save_only_single_checkpoint_default(self):
        self.config_obj.remove_option('TRAIN', 'save_only_single_checkpoint_file')
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.save_only_single_checkpoint_file is True)

    def test_checkpoint_step_default(self):
        self.config_obj.remove_option('TRAIN', 'checkpoint_step')
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.checkpoint_step is None)

    def test_patience_default(self):
        self.config_obj.remove_option('TRAIN', 'patience')
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.patience is None)

    def test_normalize_spectrograms_default(self):
        self.config_obj.remove_option('TRAIN', 'normalize_spectrograms')
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.normalize_spectrograms is False)

    def test_use_previous_run_default(self):
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.use_train_subsets_from_previous_run is False)
        self.assertTrue(train_config_tup.previous_run_path is None)

    def test_use_previous_run_without_path_error(self):
        self.config_obj['TRAIN']['use_train_subsets_from_previous_run'] = 'True'
        with self.assertRaises(KeyError):
            vak.config.train.parse_train_config(self.config_obj, self.config_file)

    def test_save_transformed_data_default(self):
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.save_transformed_data is False)


if __name__ == '__main__':
    unittest.main()

