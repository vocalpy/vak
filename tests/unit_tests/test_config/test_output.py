"""tests for vak.config.output module"""
import tempfile
import shutil
import unittest
from configparser import ConfigParser

import vak.config.output
import vak.utils


class TestParseOutputConfig(unittest.TestCase):

    def setUp(self):
        self.tmp_root_dir = tempfile.mkdtemp()
        self.tmp_results_dir = tempfile.mkdtemp(dir=self.tmp_root_dir)
        self.config_obj = ConfigParser()
        self.config_obj['OUTPUT'] = {
            'root_results_dir': str(self.tmp_root_dir),
            'results_dir_made_by_main_script': str(self.tmp_results_dir),
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_root_dir)

    def test_parse_output_config_returns_OutputConfig_instance(self):
        output_config_obj = vak.config.output.parse_output_config(self.config_obj)
        self.assertTrue(type(output_config_obj) == vak.config.output.OutputConfig)

    def test_missing_root_results_dir_raises(self):
        self.config_obj.remove_option('OUTPUT', 'root_results_dir')
        with self.assertRaises(KeyError):
            vak.config.output.parse_output_config(self.config_obj)

    def test_nonexistent_root_results_dir_raises(self):
        self.config_obj['OUTPUT']['root_results_dir'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.output.parse_output_config(self.config_obj)

    def test_no_results_dir_defaults_to_None(self):
        self.config_obj.remove_option('OUTPUT', 'results_dir_made_by_main_script')
        output_config_tup = vak.config.output.parse_output_config(self.config_obj)
        self.assertTrue(output_config_tup.results_dirname is None)


if __name__ == '__main__':
    unittest.main()
