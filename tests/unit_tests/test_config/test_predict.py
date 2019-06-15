"""tests for vak.config.predict module"""
import os
import tempfile
import shutil
import unittest
from configparser import ConfigParser
from glob import glob

import vak.config.predict
import vak.utils
from vak.core.learncurve import LEARN_CURVE_DIR_STEM

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')
TEST_CONFIGS_DIR = os.path.join(TEST_DATA_DIR, 'configs')


class TestParsePredictConfig(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()

        self.tmp_dir_to_predict = tempfile.mkdtemp()
        self.tmp_dir_to_predict = os.path.join(self.tmp_dir_to_predict, '032312')
        os.makedirs(self.tmp_dir_to_predict)

        a_results_dir = glob(os.path.join(TEST_DATA_DIR,
                                          'results',
                                          f'{LEARN_CURVE_DIR_STEM}*'))[0]
        labels_mapping_path = glob(os.path.join(a_results_dir, 'labels_mapping'))[0]
        a_training_records_dir = glob(os.path.join(a_results_dir,
                                                   'records_for_training_set*')
                                      )[0]
        checkpoint_path = os.path.join(a_training_records_dir, 'TweetyNet', 'checkpoints')
        spect_scaler = glob(os.path.join(a_training_records_dir, 'spect_scaler_*'))[0]

        # rewrite config so it points to data for testing + temporary output dirs
        a_config = os.path.join(TEST_CONFIGS_DIR, 'test_predict_config.ini')
        config = ConfigParser()
        config.read(a_config)
        config['PREDICT']['checkpoint_path'] = checkpoint_path
        config['PREDICT']['labels_mapping_path'] = labels_mapping_path
        config['PREDICT']['dir_to_predict'] = self.tmp_dir_to_predict
        config['PREDICT']['spect_scaler_path'] = spect_scaler
        self.config_obj = config

    def tearDown(self):
        shutil.rmtree(self.tmp_dir_to_predict)

    def test_parse_predict_config_returns_PredictConfig_instance(self):
        predict_config_obj = vak.config.predict.parse_predict_config(self.config_obj)
        self.assertTrue(type(predict_config_obj) == vak.config.predict.PredictConfig)

    def test_no_networks_raises(self):
        self.config_obj.remove_option('PREDICT', 'networks')
        with self.assertRaises(KeyError):
            vak.config.predict.parse_predict_config(self.config_obj)

    def test_network_not_installed_raises(self):
        self.config_obj['PREDICT']['networks'] = 'NotInstalledNet, OtherNotInstalledNet'
        with self.assertRaises(TypeError):
            vak.config.predict.parse_predict_config(self.config_obj)

    def test_missing_checkpoint_path_raises(self):
        self.config_obj.remove_option('PREDICT', 'checkpoint_path')
        with self.assertRaises(KeyError):
            vak.config.predict.parse_predict_config(self.config_obj)

    def test_missing_dir_to_predict_raises(self):
        self.config_obj.remove_option('PREDICT', 'dir_to_predict')
        with self.assertRaises(KeyError):
            vak.config.predict.parse_predict_config(self.config_obj)

    def test_nonexistent_checkpoint_dir_raises(self):
        self.config_obj['PREDICT']['checkpoint_path'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.predict.parse_predict_config(self.config_obj)

    def test_nonexistent_dir_to_predict_raises(self):
        self.config_obj['PREDICT']['dir_to_predict'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.predict.parse_predict_config(self.config_obj)


if __name__ == '__main__':
    unittest.main()
