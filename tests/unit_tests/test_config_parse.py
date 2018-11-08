"""tests for songdeck.config.parse module"""
import os
import tempfile
import shutil
import unittest
from configparser import ConfigParser

import songdeck.config
import songdeck.utils

HERE = os.path.dirname(__file__)
TEST_CONFIGS_PATH = os.path.join(HERE, '..', 'test_data', 'configs')


class TestParseSpectConfig(unittest.TestCase):

    def setUp(self):
        self.tmp_root_dir = tempfile.mkdtemp()
        self.tmp_results_dir = tempfile.mkdtemp(dir=self.tmp_root_dir)
        self.tmp_data_dir = tempfile.mkdtemp()
        self.tmp_config_dir = tempfile.mkdtemp()

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
        file_obj = tempfile.NamedTemporaryFile(prefix='config', suffix='.ini', mode='w',
                                               dir=self.tmp_config_dir, delete=False)
        with file_obj as config_file_out:
            config.write(config_file_out)
        return os.path.abspath(file_obj.name)

    def test_config_tuple_has_all_attrs(self):
        test_learncurve_config = os.path.join(TEST_CONFIGS_PATH,
                                              'test_learncurve_config.ini')
        tmp_config = self._add_dirs_to_config_and_save_as_tmp(test_learncurve_config)
        config_tup = songdeck.config.parse_config(tmp_config)
        for field in songdeck.config.parse.ConfigTuple._fields:
            self.assertTrue(hasattr(config_tup, field))


if __name__ == '__main__':
    unittest.main()
