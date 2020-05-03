"""tests for vak.config.learncurve module"""
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from configparser import ConfigParser

import vak.config.learncurve
import vak.split

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
TEST_CONFIGS_DIR = TEST_DATA_DIR.joinpath('configs')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestParseLearncurveConfig(unittest.TestCase):
    def setUp(self):
        _, self.tmp_train_vds_path = tempfile.mkstemp()
        _, self.tmp_val_vds_path = tempfile.mkstemp()
        _, self.tmp_test_vds_path = tempfile.mkstemp()
        self.tmp_root_dir = tempfile.mkdtemp()
        self.tmp_results_dir = tempfile.mkdtemp(dir=self.tmp_root_dir)

        self.config_file = TEST_CONFIGS_DIR.joinpath('test_learncurve_config.ini')
        self.config_obj = ConfigParser()
        self.config_obj.read(self.config_file)
        self.config_obj['LEARNCURVE']['train_vds_path'] = self.tmp_train_vds_path
        self.config_obj['LEARNCURVE']['val_vds_path'] = self.tmp_val_vds_path
        self.config_obj['LEARNCURVE']['test_vds_path'] = self.tmp_test_vds_path
        self.config_obj['LEARNCURVE']['root_results_dir'] = self.tmp_root_dir
        self.config_obj['LEARNCURVE']['results_dir_made_by_main_script'] = self.tmp_results_dir

    def tearDown(self):
        os.remove(self.tmp_train_vds_path)
        os.remove(self.tmp_val_vds_path)
        os.remove(self.tmp_test_vds_path)
        shutil.rmtree(self.tmp_root_dir)

    def test_parse_learncurve_config_returns_LearncurveConfig_instance(self):
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(type(learncurve_config_obj) == vak.config.learncurve.LearncurveConfig)

    def test_no_networks_raises(self):
        self.config_obj.remove_option('LEARNCURVE', 'networks')
        with self.assertRaises(KeyError):
            vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)

    def test_network_not_installed_raises(self):
        self.config_obj['LEARNCURVE']['networks'] = 'NotInstalledNet, OtherNotInstalledNet'
        with self.assertRaises(TypeError):
            vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)

    def test_no_train_path_raises(self):
        self.config_obj.remove_option('LEARNCURVE', 'train_vds_path')
        with self.assertRaises(KeyError):
            vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)

    def test_no_train_set_durs_raises(self):
        self.config_obj.remove_option('LEARNCURVE', 'train_set_durs')
        with self.assertRaises(TypeError):
            learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)

    def test_no_replicates_raises(self):
        self.config_obj.remove_option('LEARNCURVE', 'replicates')
        with self.assertRaises(TypeError):
            learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)

    def test_val_data_dict_path_default(self):
        self.config_obj.remove_option('LEARNCURVE', 'val_vds_path')
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.val_vds_path is None)

    def test_no_test_vds_path_raises(self):
        self.config_obj.remove_option('LEARNCURVE', 'test_vds_path')
        with self.assertRaises(TypeError):
            learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)

    def test_val_step_default(self):
        self.config_obj.remove_option('LEARNCURVE', 'val_step')
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.val_step is None)

    def test_save_only_single_checkpoint_default(self):
        self.config_obj.remove_option('LEARNCURVE', 'save_only_single_checkpoint_file')
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.save_only_single_checkpoint_file is True)

    def test_ckpt_step_default(self):
        self.config_obj.remove_option('LEARNCURVE', 'ckpt_step')
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.ckpt_step is None)

    def test_patience_default(self):
        self.config_obj.remove_option('LEARNCURVE', 'patience')
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.patience is None)

    def test_normalize_spectrograms_default(self):
        self.config_obj.remove_option('LEARNCURVE', 'normalize_spectrograms')
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.normalize_spectrograms is False)

    def test_use_previous_run_default(self):
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.use_train_subsets_from_previous_run is False)
        self.assertTrue(learncurve_config_obj.previous_run_path is None)

    def test_use_previous_run_without_path_error(self):
        self.config_obj['LEARNCURVE']['use_train_subsets_from_previous_run'] = 'True'
        with self.assertRaises(KeyError):
            vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)

    def test_save_transformed_data(self):
        self.config_obj['LEARNCURVE']['save_transformed_data'] = 'True'
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.save_transformed_data is True)

        self.config_obj['LEARNCURVE']['save_transformed_data'] = 'Yes'
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.save_transformed_data is True)

        self.config_obj['LEARNCURVE']['save_transformed_data'] = 'False'
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.save_transformed_data is False)

        self.config_obj['LEARNCURVE']['save_transformed_data'] = 'No'
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.save_transformed_data is False)

    def test_save_transformed_data_default(self):
        # test that save_transformed_data is added
        # and set to False, if we don't specify it
        learncurve_config_obj = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(learncurve_config_obj.save_transformed_data is False)

    def test_missing_root_results_dir_raises(self):
        self.config_obj.remove_option('LEARNCURVE', 'root_results_dir')
        with self.assertRaises(KeyError):
            vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)

    def test_nonexistent_root_results_dir_raises(self):
        self.config_obj['LEARNCURVE']['root_results_dir'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)

    def test_no_results_dir_defaults_to_None(self):
        self.config_obj.remove_option('LEARNCURVE', 'results_dir_made_by_main_script')
        train_config_tup = vak.config.learncurve.parse_learncurve_config(self.config_obj, self.config_file)
        self.assertTrue(train_config_tup.results_dirname is None)


if __name__ == '__main__':
    unittest.main()
