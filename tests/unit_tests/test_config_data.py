import os
import unittest
from configparser import ConfigParser

import songdeck.config.data
import songdeck.utils

HERE = os.path.dirname(__file__)


def _base_config():
    base_config = ConfigParser()
    test_data_dir = os.path.join(HERE,
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

    def test_str_labelset(self):
        config_obj = self.get_config

        config_file = 'test'
        data_config_tup = songdeck.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertEqual(data_config_tup.labelset,
                         list(config_obj['DATA']['labelset'])
                         )

    def test_rangestr_labelset(self):
        a_rangestr = '1-9, 12'
        config_obj = self.get_config
        config_obj['DATA']['labelset'] = a_rangestr

        config_file = 'test'
        data_config_tup = songdeck.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertEqual(data_config_tup.labelset,
                         songdeck.utils.range_str(a_rangestr)
                         )

    def test_int_labelset(self):
        int_labels = '01234567'
        config_obj = self.get_config
        config_obj['DATA']['labelset'] = int_labels

        config_file = 'test'
        data_config_tup = songdeck.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertEqual(data_config_tup.labelset,
                         list(int_labels)
                         )

    def test_all_labels_are_int_default(self):
        # test that all_labels_are_int is added
        # and set to False if we don't specify it
        config_obj = self.get_config
        config_file = 'test'
        data_config_tup = songdeck.config.data.parse_data_config(config_obj,
                                                                 config_file)
        self.assertTrue(hasattr(data_config_tup,'all_labels_are_int'))
        self.assertTrue(data_config_tup.all_labels_are_int is False)


if __name__ == '__main__':
    unittest.main()
