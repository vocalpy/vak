"""tests for vak.config.train module"""
import unittest
import os
import shutil
import tempfile
from configparser import ConfigParser

import vak.config.train
import vak.split


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')
TEST_CONFIGS_DIR = os.path.join(TEST_DATA_DIR, 'configs')


class TestParseTrainConfig(unittest.TestCase):
    def setUp(self):
        _, self.tmp_train_vds_path = tempfile.mkstemp()
        _, self.tmp_val_vds_path = tempfile.mkstemp()
        self.tmp_root_dir = tempfile.mkdtemp()
        self.tmp_results_dir = tempfile.mkdtemp(dir=self.tmp_root_dir)

        self.config_file = os.path.join(TEST_DATA_DIR, 'configs', 'test_train_config.ini')
        self.config_obj = ConfigParser()
        self.config_obj.read(self.config_file)
        self.config_obj['TRAIN']['train_vds_path'] = self.tmp_train_vds_path
        self.config_obj['TRAIN']['val_vds_path'] = self.tmp_val_vds_path
        self.config_obj['TRAIN']['root_results_dir'] = self.tmp_root_dir
        self.config_obj['TRAIN']['results_dir_made_by_main_script'] = self.tmp_results_dir

    def tearDown(self):
        os.remove(self.tmp_train_vds_path)
        os.remove(self.tmp_val_vds_path)
        shutil.rmtree(self.tmp_root_dir)

    def test_parse_train_config_returns_TrainConfig_instance(self):
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(type(train_config_obj) == vak.config.train.TrainConfig)

    def test_no_networks_raises(self):
        self.config_obj.remove_option('TRAIN', 'networks')
        with self.assertRaises(KeyError):
            vak.config.train.parse_train_config(self.config_obj, self.config_file)

    def test_network_not_installed_raises(self):
        self.config_obj['TRAIN']['networks'] = 'NotInstalledNet, OtherNotInstalledNet'
        with self.assertRaises(TypeError):
            vak.config.train.parse_train_config(self.config_obj, self.config_file)

    def test_no_train_path_raises(self):
        self.config_obj.remove_option('TRAIN', 'train_vds_path')
        with self.assertRaises(KeyError):
            vak.config.train.parse_train_config(self.config_obj, self.config_file)

    def test_val_data_dict_path_default(self):
        self.config_obj.remove_option('TRAIN', 'val_vds_path')
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.val_vds_path is None)

    def test_val_step_default(self):
        self.config_obj.remove_option('TRAIN', 'val_step')
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.val_step is None)

    def test_save_only_single_checkpoint_default(self):
        self.config_obj.remove_option('TRAIN', 'save_only_single_checkpoint_file')
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.save_only_single_checkpoint_file is True)

    def test_ckpt_step_default(self):
        self.config_obj.remove_option('TRAIN', 'ckpt_step')
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.ckpt_step is None)

    def test_patience_default(self):
        self.config_obj.remove_option('TRAIN', 'patience')
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.patience is None)

    def test_normalize_spectrograms_default(self):
        self.config_obj.remove_option('TRAIN', 'normalize_spectrograms')
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.normalize_spectrograms is False)

    def test_use_previous_run_default(self):
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.use_train_subsets_from_previous_run is False)
        self.assertTrue(train_config_obj.previous_run_path is None)

    def test_use_previous_run_without_path_error(self):
        self.config_obj['TRAIN']['use_train_subsets_from_previous_run'] = 'True'
        with self.assertRaises(KeyError):
            vak.config.train.parse_train_config(self.config_obj, self.config_file)

    def test_save_transformed_data(self):
        self.config_obj['TRAIN']['save_transformed_data'] = 'True'
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.save_transformed_data is True)

        self.config_obj['TRAIN']['save_transformed_data'] = 'Yes'
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.save_transformed_data is True)

        self.config_obj['TRAIN']['save_transformed_data'] = 'False'
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.save_transformed_data is False)

        self.config_obj['TRAIN']['save_transformed_data'] = 'No'
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.save_transformed_data is False)

    def test_save_transformed_data_default(self):
        # test that save_transformed_data is added
        # and set to False, if we don't specify it
        train_config_obj = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_obj.save_transformed_data is False)

    def test_missing_root_results_dir_raises(self):
        self.config_obj.remove_option('TRAIN', 'root_results_dir')
        with self.assertRaises(KeyError):
            vak.config.train.parse_train_config(self.config_obj, self.config_file)

    def test_nonexistent_root_results_dir_raises(self):
        self.config_obj['TRAIN']['root_results_dir'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.train.parse_train_config(self.config_obj, self.config_file)

    def test_no_results_dir_defaults_to_None(self):
        self.config_obj.remove_option('TRAIN', 'results_dir_made_by_main_script')
        train_config_tup = vak.config.train.parse_train_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.results_dirname is None)


if __name__ == '__main__':
    unittest.main()
