"""tests for vak.config.prep module"""
from configparser import ConfigParser
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import vak.config.converters
import vak.config.prep
import vak.split

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
TEST_CONFIGS_DIR = TEST_DATA_DIR.joinpath('configs')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestParsePrepConfig(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()

        test_learncurve_config = TEST_CONFIGS_DIR.joinpath('test_learncurve_config.ini')
        # Now we want a copy (of the changed version) to use for tests
        # since this is what the test data was made with
        self.tmp_config_dir = tempfile.mkdtemp()
        self.config_file = Path(self.tmp_config_dir).joinpath('tmp_test_learncurve_config.ini')
        shutil.copy(test_learncurve_config, self.config_file)

        # rewrite config so it points to existing directories
        config = ConfigParser()
        config.read(self.config_file)

        config['PREP']['data_dir'] = str(TEST_DATA_DIR.joinpath('cbins', 'gy6or6', '032312'))
        config['PREP']['output_dir'] = str(self.tmp_output_dir)
        with open(self.config_file, 'w') as fp:
            config.write(fp)

        self.config_obj = config

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        shutil.rmtree(self.tmp_config_dir)

    def test_parse_prep_config_returns_DataConfig_instance(self):
        prep_config_obj = vak.config.prep.parse_prep_config(self.config_obj, self.config_file)
        self.assertTrue(type(prep_config_obj) == vak.config.prep.PrepConfig)

    def test_no_data_dir_raises(self):
        self.config_obj.remove_option('PREP', 'data_dir')
        with self.assertRaises(TypeError):
            prep_config_obj = vak.config.prep.parse_prep_config(self.config_obj, self.config_file)

    def test_nonexistent_data_dir_raises_error(self):
        # test that mate_spect_files_path is added
        # and set to None if we don't specify it
        self.config_obj['PREP']['data_dir'] = 'theres/no/way/this/is/a/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.prep.parse_prep_config(self.config_obj, self.config_file)

    def test_no_output_dir_raises(self):
        if self.config_obj.has_option('PREP', 'output_dir'):
            self.config_obj.remove_option('PREP', 'output_dir')
        with self.assertRaises(TypeError):
            prep_config_obj = vak.config.prep.parse_prep_config(self.config_obj, self.config_file)

    def test_str_labelset(self):
        prep_config_obj = vak.config.prep.parse_prep_config(self.config_obj, self.config_file)
        self.assertEqual(
            prep_config_obj.labelset, list(self.config_obj['PREP']['labelset'])
        )

    def test_rangestr_labelset(self):
        a_rangestr = '1-9, 12'
        self.config_obj['PREP']['labelset'] = a_rangestr
        prep_config_obj = vak.config.prep.parse_prep_config(self.config_obj, self.config_file)
        self.assertEqual(
            prep_config_obj.labelset, vak.config.converters.range_str(a_rangestr)
        )

    def test_int_labelset(self):
        int_labels = '01234567'
        self.config_obj['PREP']['labelset'] = int_labels
        prep_config_obj = vak.config.prep.parse_prep_config(self.config_obj, self.config_file)
        self.assertEqual(
            prep_config_obj.labelset, list(int_labels)
        )

    def test_both_audio_and_spect_format_raises(self):
        # learncurve_config already has 'audio_format', if we
        # also add spect_format, should raise an error
        self.config_obj['PREP']['spect_format'] = 'mat'
        with self.assertRaises(ValueError):
            vak.config.prep.parse_prep_config(self.config_obj, self.config_file)

    def test_neither_audio_nor_spect_format_raises(self):
        # if we remove audio_format option, then neither that or
        # spect_format is specified, should raise an error
        self.config_obj.remove_option('PREP','audio_format')
        with self.assertRaises(ValueError):
            vak.config.prep.parse_prep_config(self.config_obj, self.config_file)


if __name__ == '__main__':
    unittest.main()
