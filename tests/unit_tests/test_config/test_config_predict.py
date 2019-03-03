"""tests for vak.config.predict module"""
import os
import tempfile
import shutil
import unittest
from configparser import ConfigParser

import vak.config.predict
import vak.utils

HERE = os.path.dirname(__file__)
TEST_CONFIGS_PATH = os.path.join(HERE, '..', 'test_data', 'configs')


def _base_config(tmp_checkpoint_dir,
                 tmp_dir_to_predict):
    base_config = ConfigParser()
    base_config['PREDICT'] = {
        'networks': 'VakTestNet',
        'checkpoint_dir': str(tmp_checkpoint_dir),
        'dir_to_predict': str(tmp_dir_to_predict),
    }
    return base_config


class TestParsePredictConfig(unittest.TestCase):

    def setUp(self):
        self.tmp_checkpoint_dir = tempfile.mkdtemp()
        self.tmp_dir_to_predict = tempfile.mkdtemp()
        self.get_config = _base_config(self.tmp_checkpoint_dir,
                                       self.tmp_dir_to_predict)

    def tearDown(self):
        shutil.rmtree(self.tmp_checkpoint_dir)
        shutil.rmtree(self.tmp_dir_to_predict)

    def test_config_tuple_has_all_attrs(self):
        config_obj = self.get_config
        predict_config_tup = vak.config.predict.parse_predict_config(config_obj)
        for field in vak.config.predict.PredictConfig._fields:
            self.assertTrue(hasattr(predict_config_tup, field))

    def test_no_networks_raises(self):
        config_obj = self.get_config
        config_obj.remove_option('PREDICT', 'networks')
        with self.assertRaises(KeyError):
            vak.config.predict.parse_predict_config(config_obj)

    def test_network_not_installed_raises(self):
        config_obj = self.get_config
        config_obj['PREDICT']['networks'] = 'NotInstalledNet, OtherNotInstalledNet'
        with self.assertRaises(TypeError):
            vak.config.predict.parse_predict_config(config_obj)

    def test_missing_checkpoint_dir_raises(self):
        config_obj = self.get_config
        config_obj.remove_option('PREDICT', 'checkpoint_dir')
        with self.assertRaises(KeyError):
            vak.config.predict.parse_predict_config(config_obj)

    def test_missing_dir_to_predict_raises(self):
        config_obj = self.get_config
        config_obj.remove_option('PREDICT', 'dir_to_predict')
        with self.assertRaises(KeyError):
            vak.config.predict.parse_predict_config(config_obj)

    def test_nonexistent_checkpoint_dir_raises(self):
        config_obj = self.get_config
        config_obj['PREDICT']['checkpoint_dir'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.predict.parse_predict_config(config_obj)

    def test_nonexistent_dir_to_predict_raises(self):
        config_obj = self.get_config
        config_obj['PREDICT']['dir_to_predict'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.predict.parse_predict_config(config_obj)


if __name__ == '__main__':
    unittest.main()
