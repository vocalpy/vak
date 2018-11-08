"""tests for songdeck.config.output module"""
import tempfile
import shutil
import unittest
from configparser import ConfigParser

import songdeck.config.output
import songdeck.utils


def _base_config(tmp_root_dir,
                 tmp_results_dir):
    base_config = ConfigParser()
    base_config['OUTPUT'] = {
        'root_results_dir': str(tmp_root_dir),
        'results_dir_made_by_main_script': str(tmp_results_dir),
    }
    return base_config


class TestParseOutputConfig(unittest.TestCase):

    def setUp(self):
        self.tmp_root_dir = tempfile.mkdtemp()
        self.tmp_results_dir = tempfile.mkdtemp(dir=self.tmp_root_dir)
        self.get_config = _base_config(self.tmp_root_dir,
                                       self.tmp_results_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_root_dir)

    def test_config_tuple_has_all_attrs(self):
        config_obj = self.get_config
        output_config_tup = songdeck.config.output.parse_output_config(config_obj)
        for field in songdeck.config.output.OutputConfig._fields:
            self.assertTrue(hasattr(output_config_tup, field))

    def test_missing_root_results_dir_raises(self):
        config_obj = self.get_config
        config_obj.remove_option('OUTPUT', 'root_results_dir')
        with self.assertRaises(KeyError):
            songdeck.config.output.parse_output_config(config_obj)

    def test_nonexistent_root_results_dir_raises(self):
        config_obj = self.get_config
        config_obj['OUTPUT']['root_results_dir'] = 'obviously/non/existent/dir'
        with self.assertRaises(NotADirectoryError):
            songdeck.config.output.parse_output_config(config_obj)

    def test_no_results_dir_defaults_to_None(self):
        config_obj = self.get_config
        config_obj.remove_option('OUTPUT', 'results_dir_made_by_main_script')
        output_config_tup = songdeck.config.output.parse_output_config(config_obj)
        self.assertTrue(output_config_tup.results_dirname is None)


if __name__ == '__main__':
    unittest.main()
