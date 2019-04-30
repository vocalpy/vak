"""tests for vak.config.data module"""
import os
import unittest
from configparser import ConfigParser
import tempfile
import shutil

import vak.config.data
import vak.utils

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')
TEST_CONFIGS_DIR = os.path.join(TEST_DATA_DIR, 'configs')


class TestParseDataConfig(unittest.TestCase):
    def setUp(self):
        self.config_file = os.path.join(TEST_DATA_DIR, 'configs', 'test_learncurve_config.ini')
        self.config_obj = ConfigParser()
        self.config_obj.read(self.config_file)

        self.tmp_data_dir = tempfile.mkdtemp()
        self.config_obj['DATA']['data_dir'] = self.tmp_data_dir

    def tearDown(self):
        shutil.rmtree(self.tmp_data_dir)

    def test_parse_data_config_returns_DataConfig_instance(self):
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertTrue(type(data_config_obj) == vak.config.data.DataConfig)

    def test_str_labelset(self):
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertEqual(
            data_config_obj.labelset, list(self.config_obj['DATA']['labelset'])
        )

    def test_rangestr_labelset(self):
        a_rangestr = '1-9, 12'
        self.config_obj['DATA']['labelset'] = a_rangestr
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertEqual(
            data_config_obj.labelset, vak.utils.data.range_str(a_rangestr)
        )

    def test_int_labelset(self):
        int_labels = '01234567'
        self.config_obj['DATA']['labelset'] = int_labels
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertEqual(
            data_config_obj.labelset, list(int_labels)
        )

    def test_all_labels_are_int_default(self):
        # test that all_labels_are_int is added
        # and set to False if we don't specify it
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertTrue(data_config_obj.all_labels_are_int is False)

    def test_silent_gap_label_default(self):
        # test that silent_gap_label is added
        # and set to 0 if we don't specify it
        if self.config_obj.has_option('DATA', 'silent_gap_label'):
            self.config_obj.remove_option('DATA', 'silent_gap_label')
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertTrue(data_config_obj.silent_gap_label == 0)

    def test_output_dir_default(self):
        # test that output_dir is added
        # and set to None if we don't specify it
        if self.config_obj.has_option('DATA', 'output_dir'):
            self.config_obj.remove_option('DATA', 'output_dir')
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertTrue(data_config_obj.output_dir is None)

    def test_both_audio_and_spect_format_raises(self):
        # learncurve_config already has 'audio_format', if we
        # also add spect_format, should raise an error
        self.config_obj['DATA']['spect_format'] = 'mat'
        with self.assertRaises(ValueError):
            vak.config.data.parse_data_config(self.config_obj, self.config_file)

    def test_neither_audio_nor_spect_format_raises(self):
        # if we remove audio_format option, then neither that or
        # spect_format is specified, should raise an error
        self.config_obj.remove_option('DATA','audio_format')
        with self.assertRaises(ValueError):
            vak.config.data.parse_data_config(self.config_obj, self.config_file)

    def test_data_dir_default(self):
        # test that data_dir set to None if we don't specify it
        self.config_obj.remove_option('DATA', 'data_dir')
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertTrue(data_config_obj.data_dir is None)

    def test_nonexistent_data_dir_raises_error(self):
        # test that mate_spect_files_path is added
        # and set to None if we don't specify it
        self.config_obj['DATA']['data_dir'] = 'theres/no/way/this/is/a/dir'
        with self.assertRaises(NotADirectoryError):
            vak.config.data.parse_data_config(self.config_obj, self.config_file)

    def test_save_transformed_data(self):
        self.config_obj['DATA']['save_transformed_data'] = 'True'
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertTrue(data_config_obj.save_transformed_data is True)

        self.config_obj['DATA']['save_transformed_data'] = 'Yes'
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertTrue(data_config_obj.save_transformed_data is True)

        self.config_obj['DATA']['save_transformed_data'] = 'False'
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertTrue(data_config_obj.save_transformed_data is False)

        self.config_obj['DATA']['save_transformed_data'] = 'No'
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertTrue(data_config_obj.save_transformed_data is False)

    def test_save_transformed_data_default(self):
        # test that save_transformed_data is added
        # and set to False, if we don't specify it
        data_config_obj = vak.config.data.parse_data_config(self.config_obj, self.config_file)
        self.assertTrue(data_config_obj.save_transformed_data is False)


if __name__ == '__main__':
    unittest.main()
