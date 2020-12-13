"""tests for vak.config.predict module"""
from configparser import ConfigParser
from pathlib import Path
import unittest

import vak.config.predict
import vak.split
from vak.core.learncurve import LEARN_CURVE_DIR_STEM

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
TEST_CONFIGS_PATH = TEST_DATA_DIR.joinpath('configs')


class TestParsePredictConfig(unittest.TestCase):
    def setUp(self):
        a_results_dir = list(
            TEST_DATA_DIR.joinpath('results').glob(
                f'{LEARN_CURVE_DIR_STEM}*'))[0]
        a_training_records_dir = list(
            Path(a_results_dir).joinpath(
                'train').glob('records_for_training_set*'))[0]
        checkpoint_path = str(Path(a_training_records_dir).joinpath(
            'TweetyNet', 'checkpoints'))
        spect_scaler = list(
            Path(a_training_records_dir).glob('spect_scaler_*'))[0]
        spect_scaler = str(spect_scaler)

        # rewrite config so it points to data for testing + temporary output dirs
        a_config = str(TEST_CONFIGS_PATH.joinpath('test_predict_config.ini'))
        config = ConfigParser()
        config.read(a_config)
        config['PREDICT']['checkpoint_path'] = checkpoint_path
        config['PREDICT']['spect_scaler_path'] = spect_scaler
        test_data_vds_path = list(TEST_DATA_DIR.glob('vds'))[0]
        test_data_vds_path = Path(test_data_vds_path)
        for stem in ['train', 'test']:
            vds_path = list(test_data_vds_path.glob(f'*.{stem}.vds.json'))
            self.assertTrue(len(vds_path) == 1)
            vds_path = vds_path[0]
            if stem == 'train':
                config['PREDICT']['train_vds_path'] = str(vds_path)
            elif stem == 'test':
                # pretend test data is data we want to predict
                config['PREDICT']['predict_vds_path'] = str(vds_path)

        self.config_obj = config

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

    def test_missing_predict_vds_path_raises(self):
        self.config_obj.remove_option('PREDICT', 'predict_vds_path')
        with self.assertRaises(KeyError):
            vak.config.predict.parse_predict_config(self.config_obj)

    def test_missing_train_vds_path_raises(self):
        self.config_obj.remove_option('PREDICT', 'train_vds_path')
        with self.assertRaises(KeyError):
            vak.config.predict.parse_predict_config(self.config_obj)

    def test_nonexistent_checkpoint_dir_raises(self):
        self.config_obj['PREDICT']['checkpoint_path'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.predict.parse_predict_config(self.config_obj)

    def test_nonexistent_predict_vds_path_raises(self):
        self.config_obj['PREDICT']['predict_vds_path'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.predict.parse_predict_config(self.config_obj)

    def test_nonexistent_train_vds_path_raises(self):
        self.config_obj['PREDICT']['train_vds_path'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.predict.parse_predict_config(self.config_obj)


if __name__ == '__main__':
    unittest.main()
