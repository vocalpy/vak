"""tests for vak.config.data module"""
import os
import unittest
from configparser import ConfigParser

import vak.config.data
import vak.utils

HERE = os.path.dirname(__file__)


def _base_config():
    base_config = ConfigParser()
    test_data_dir = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'test_data')
    base_config['DATA'] = {
        'labelset': 'iabcdefghjk',
        'data_dir': test_data_dir,
        'total_train_set_duration': '400',
        'validation_set_duration': '100',
        'test_set_duration': '400',
        'skip_files_with_labels_not_in_labelset': 'Yes',
    }
    return base_config


class TestParseDataConfig(unittest.TestCase):

    def setUp(self):
        self.get_config = _base_config()

    def test_config_tuple_has_all_attrs(self):
        config_obj = self.get_config
        config_file = 'test'
        data_config_instance = vak.config.data.parse_data_config(config_obj, config_file)
        data_config_attrs = [attr.name for attr in vak.config.data.DataConfig.__attrs_attrs__]
        for data_config_attr in data_config_attrs:
            self.assertTrue(hasattr(data_config_instance, data_config_attr))

    def test_str_labelset(self):
        config_obj = self.get_config

        config_file = 'test'
        data_config_tup = vak.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertEqual(data_config_tup.labelset,
                         list(config_obj['DATA']['labelset'])
                         )

    def test_rangestr_labelset(self):
        a_rangestr = '1-9, 12'
        config_obj = self.get_config
        config_obj['DATA']['labelset'] = a_rangestr

        config_file = 'test'
        data_config_tup = vak.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertEqual(data_config_tup.labelset,
                         vak.utils.data.range_str(a_rangestr)
                         )

    def test_int_labelset(self):
        int_labels = '01234567'
        config_obj = self.get_config
        config_obj['DATA']['labelset'] = int_labels

        config_file = 'test'
        data_config_tup = vak.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertEqual(data_config_tup.labelset,
                         list(int_labels)
                         )

    def test_all_labels_are_int_default(self):
        # test that all_labels_are_int is added
        # and set to False if we don't specify it
        config_obj = self.get_config
        config_file = 'test'
        data_config_tup = vak.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertTrue(data_config_tup.all_labels_are_int is False)

    def test_silent_gap_label_default(self):
        # test that silent_gap_label is added
        # and set to 0 if we don't specify it
        config_obj = self.get_config
        config_file = 'test'
        data_config_tup = vak.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertTrue(data_config_tup.silent_gap_label == 0)

    def test_output_dir_default(self):
        # test that output_dir is added
        # and set to None if we don't specify it
        config_obj = self.get_config
        config_file = 'test'
        data_config_tup = vak.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertTrue(data_config_tup.output_dir is None)

    def test_mat_spect_files_path_default(self):
        # test that mat_spect_files_path is added
        # and set to None if we don't specify it
        config_obj = self.get_config
        config_file = 'test'
        data_config_tup = vak.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertTrue(data_config_tup.mat_spect_files_path is None)

    def test_nonexistent_data_dir_raises_error(self):
        # test that mate_spect_files_path is added
        # and set to None if we don't specify it
        config_obj = self.get_config
        config_obj['DATA']['data_dir'] = 'theres/no/way/this/is/a/dir'
        config_file = 'test'
        with self.assertRaises(NotADirectoryError):
            vak.config.data.parse_data_config(config_obj, config_file)

    def test_save_transformed_data(self):
        # test that save_transformed_data is added
        # and set to False, if we don't specify it
        config_obj = self.get_config
        config_obj['DATA']['save_transformed_data'] = 'True'
        config_file = 'test'
        data_config_tup = vak.config.data.parse_data_config(config_obj, config_file)
        self.assertTrue(data_config_tup.save_transformed_data is True)

        config_obj['DATA']['save_transformed_data'] = 'Yes'
        data_config_tup = vak.config.data.parse_data_config(config_obj, config_file)
        self.assertTrue(data_config_tup.save_transformed_data is True)

        config_obj['DATA']['save_transformed_data'] = 'False'
        data_config_tup = vak.config.data.parse_data_config(config_obj, config_file)
        self.assertTrue(data_config_tup.save_transformed_data is False)

        config_obj['DATA']['save_transformed_data'] = 'No'
        data_config_tup = vak.config.data.parse_data_config(config_obj, config_file)
        self.assertTrue(data_config_tup.save_transformed_data is False)

    def test_save_transformed_data_default(self):
        # test that save_transformed_data is added
        # and set to False, if we don't specify it
        config_obj = self.get_config
        config_file = 'test'
        data_config_tup = vak.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertTrue(data_config_tup.save_transformed_data is False)


if __name__ == '__main__':
    unittest.main()
