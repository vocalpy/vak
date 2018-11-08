"""tests for songdeck.config.train module"""
import unittest
from configparser import ConfigParser, NoOptionError

import songdeck.config.train
import songdeck.utils


def _base_config():
    base_config = ConfigParser()
    base_config['TRAIN'] = {
        'train_data_path': '/home/user/data/subdir/subsubdir1/spects/train_data_dict',
        'val_data_path ': '/home/user/data/subdir/subsubdir1/spects/val_data_dict',
        'test_data_path': '/home/user/data/subdir/subsubdir1/spects/test_data_dict',
        'normalize_spectrograms': 'Yes',
        'train_set_durs': '5, 15, 30, 45, 60, 75, 90, 105',
        'n_max_iter': '18000',
        'val_error_step': '150',
        'checkpoint_step': '600',
        'save_only_single_checkpoint_file': 'True',
        'patience': 'None',
        'replicates': '5',
    }
    return base_config


class TestParseTrainConfig(unittest.TestCase):

    def setUp(self):
        self.get_config = _base_config()

    def test_config_tuple_has_all_attrs(self):
        config_obj = self.get_config
        config_file = 'test'
        train_config_tup = songdeck.config.train.parse_train_config(config_obj, config_file)
        for field in songdeck.config.train.TrainConfig._fields:
            self.assertTrue(hasattr(train_config_tup, field))

    def test_save_only_single_checkpoint_default(self):
        config_obj = self.get_config
        config_file = 'test'
        config_obj.remove_option('TRAIN', 'save_only_single_checkpoint_file')
        train_config_tup = songdeck.config.train.parse_train_config(config_obj, config_file)
        self.assertTrue(train_config_tup.save_only_single_checkpoint_file is True)

    def test_val_error_step_default(self):
        config_obj = self.get_config
        config_file = 'test'
        config_obj.remove_option('TRAIN', 'val_error_step')
        train_config_tup = songdeck.config.train.parse_train_config(config_obj, config_file)
        self.assertTrue(train_config_tup.val_error_step is None)

    def test_checkpoint_step_default(self):
        config_obj = self.get_config
        config_file = 'test'
        config_obj.remove_option('TRAIN', 'checkpoint_step')
        train_config_tup = songdeck.config.train.parse_train_config(config_obj, config_file)
        self.assertTrue(train_config_tup.checkpoint_step is None)

    def test_patience_default(self):
        config_obj = self.get_config
        config_file = 'test'
        config_obj.remove_option('TRAIN', 'patience')
        train_config_tup = songdeck.config.train.parse_train_config(config_obj, config_file)
        self.assertTrue(train_config_tup.patience is None)

    def test_normalize_spectrograms_default(self):
        config_obj = self.get_config
        config_file = 'test'
        config_obj.remove_option('TRAIN', 'normalize_spectrograms')
        train_config_tup = songdeck.config.train.parse_train_config(config_obj, config_file)
        self.assertTrue(train_config_tup.normalize_spectrograms is False)

    def test_use_previous_run_default(self):
        config_obj = self.get_config
        config_file = 'test'
        train_config_tup = songdeck.config.train.parse_train_config(config_obj, config_file)
        self.assertTrue(train_config_tup.use_train_subsets_from_previous_run is False)
        self.assertTrue(train_config_tup.previous_run_path is None)

    def test_use_previous_run_without_path_error(self):
        config_obj = self.get_config
        config_obj['TRAIN']['use_train_subsets_from_previous_run'] = 'Yes'
        config_file = 'test'
        with self.assertRaises(KeyError):
            songdeck.config.train.parse_train_config(config_obj, config_file)


if __name__ == '__main__':
    unittest.main()

