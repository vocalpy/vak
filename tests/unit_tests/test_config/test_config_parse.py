"""tests for songdeck.config.parse module"""
import os
from glob import glob
import tempfile
import shutil
import unittest
from configparser import ConfigParser

import songdeck.config
import songdeck.utils

HERE = os.path.dirname(__file__)
TEST_CONFIGS_PATH = os.path.join(HERE, '..', '..', 'test_data', 'configs')


class TestParseConfig(unittest.TestCase):

    def setUp(self):
        self.tmp_root_dir = tempfile.mkdtemp()
        self.tmp_results_dir = tempfile.mkdtemp(dir=self.tmp_root_dir)
        self.tmp_data_dir = tempfile.mkdtemp()
        self.tmp_config_dir = tempfile.mkdtemp()
        self.tmp_checkpoint_dir = tempfile.mkdtemp()
        self.tmp_dir_to_predict = tempfile.mkdtemp()
        self.section_to_field_map = {
            'DATA': 'data',
            'SPECTROGRAM': 'spect_params',
            'TRAIN': 'train',
            'OUTPUT': 'output',
            'PREDICT': 'predict',
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_root_dir)
        shutil.rmtree(self.tmp_data_dir)

    def _add_dirs_to_config_and_save_as_tmp(self, config_file):
        config = ConfigParser()
        config.read(config_file)
        if config.has_section('OUTPUT'):
            config['OUTPUT']['root_results_dir'] = self.tmp_root_dir
            config['OUTPUT']['results_dir_made_by_main_script'] = self.tmp_results_dir
        if config.has_section('DATA'):
            config['DATA']['data_dir'] = self.tmp_data_dir
        if config.has_section('PREDICT'):
            config['PREDICT']['checkpoint_dir'] = self.tmp_checkpoint_dir
            config['PREDICT']['dir_to_predict'] = self.tmp_dir_to_predict
        file_obj = tempfile.NamedTemporaryFile(prefix='config', suffix='.ini', mode='w',
                                               dir=self.tmp_config_dir, delete=False)
        with file_obj as config_file_out:
            config.write(config_file_out)
        return os.path.abspath(file_obj.name)

    def test_config_tuple_has_all_fields(self):
        # shouldn't matter which config we use, because all fields
        # should be present no matter what (some may default to None
        # if corresponding section not defined in config.ini file)
        test_learncurve_config = os.path.join(TEST_CONFIGS_PATH,
                                              'test_learncurve_config.ini')
        tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_learncurve_config)
        config_tup = songdeck.config.parse_config(tmp_config_file)
        for field in songdeck.config.parse.ConfigTuple._fields:
            self.assertTrue(hasattr(config_tup, field))

    def test_defined_sections_not_None(self):
        test_configs = glob(os.path.join(TEST_CONFIGS_PATH,
                                         'test_*_config.ini'))
        for test_config in test_configs:
            tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_config)
            config_obj = ConfigParser()
            config_obj.read(tmp_config_file)
            config_tup = songdeck.config.parse_config(tmp_config_file)
            for section in config_obj.sections():
                field = self.section_to_field_map[section]
                self.assertTrue(getattr(config_tup, field) is not None)

    def test_both_train_and_predict_raises(self):
        test_learncurve_config = os.path.join(TEST_CONFIGS_PATH,
                                              'test_learncurve_config.ini')
        tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_learncurve_config)
        config = ConfigParser()
        config.read(tmp_config_file)
        config.add_section('PREDICT')
        config['PREDICT']['networks'] = 'SongdeckTestNet'
        config['PREDICT']['checkpoint_dir'] = self.tmp_checkpoint_dir
        config['PREDICT']['dir_to_predict'] = self.tmp_dir_to_predict
        with open(tmp_config_file, 'w') as rewrite:
            config.write(rewrite)
        with self.assertRaises(ValueError):
            songdeck.config.parse_config(tmp_config_file)


if __name__ == '__main__':
    unittest.main()
