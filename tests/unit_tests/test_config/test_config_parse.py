"""tests for vak.config.parse module"""
import os
from glob import glob
import tempfile
import shutil
import unittest
from configparser import ConfigParser
import pickle

import numpy as np
import joblib

import vak.config
import vak.utils
import vak.network
import vak.utils.data

HERE = os.path.dirname(__file__)
TEST_CONFIGS_PATH = os.path.join(HERE, '..', '..', 'test_data', 'configs')


class TestParseConfig(unittest.TestCase):

    def setUp(self):
        # for output section of config
        self.tmp_root_dir = tempfile.mkdtemp()
        self.tmp_results_dir = tempfile.mkdtemp(dir=self.tmp_root_dir)
        self.tmp_data_dir = tempfile.mkdtemp()
        self.tmp_config_dir = tempfile.mkdtemp()

        # for predict section of config
        self.tmp_checkpoint_dir = tempfile.mkdtemp()
        self.tmp_dir_to_predict = tempfile.mkdtemp()
        labels_mapping = dict(zip([int(label) for label in [int(char) for char in '1234567']],
                                  range(1, len('1234567') + 1)))
        labels_mapping_file = os.path.join(self.tmp_results_dir, 'labels_mapping')
        with open(labels_mapping_file, 'wb') as labels_map_file_obj:
            pickle.dump(labels_mapping, labels_map_file_obj)
        self.tmp_labels_mapping_path = labels_mapping_file
        a_spect_scaler = vak.utils.data.SpectScaler()
        a_spect_scaler.fit(np.random.normal(size=(1000, 513)))
        tmp_spect_scaler_path = os.path.join(self.tmp_results_dir, 'spect_scaler')
        joblib.dump(value=a_spect_scaler, filename=tmp_spect_scaler_path)
        self.tmp_spect_scaler_path = tmp_spect_scaler_path

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
        """helper functions called by unit tests to add directories
        that actually exist to avoid spurious NotADirectory errors"""
        config = ConfigParser()
        config.read(config_file)
        if config.has_section('OUTPUT'):
            config['OUTPUT']['root_results_dir'] = self.tmp_root_dir
            config['OUTPUT']['results_dir_made_by_main_script'] = self.tmp_results_dir
        if config.has_section('DATA'):
            config['DATA']['data_dir'] = self.tmp_data_dir
        if config.has_section('PREDICT'):
            config['PREDICT']['checkpoint_path'] = self.tmp_checkpoint_dir
            config['PREDICT']['dir_to_predict'] = self.tmp_dir_to_predict
            config['PREDICT']['labels_mapping_path'] = self.tmp_labels_mapping_path
            config['PREDICT']['spect_scaler_path'] = self.tmp_spect_scaler_path

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
        config_tup = vak.config.parse_config(tmp_config_file)
        for field in vak.config.parse.ConfigTuple._fields:
            self.assertTrue(hasattr(config_tup, field))

    def test_defined_sections_not_None(self):
        test_configs = glob(os.path.join(TEST_CONFIGS_PATH,
                                         'test_*_config.ini'))
        for test_config in test_configs:
            tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_config)
            config_obj = ConfigParser()
            config_obj.read(tmp_config_file)
            config_tup = vak.config.parse_config(tmp_config_file)
            for section in config_obj.sections():
                if section in self.section_to_field_map:
                    # check sections that any config.ini file can have, non-network specific
                    field = self.section_to_field_map[section]
                    self.assertTrue(getattr(config_tup, field) is not None)
                elif section in config_tup.networks:
                    # check network specific sections
                    self.assertTrue(getattr(config_tup.networks, field) is not None)

    def test_network_sections_match_config(self):
        test_configs = glob(os.path.join(TEST_CONFIGS_PATH,
                                         'test_*_config.ini'))
        NETWORKS = vak.network._load()
        for test_config in test_configs:
            tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_config)
            config_obj = ConfigParser()
            config_obj.read(tmp_config_file)
            config_tup = vak.config.parse_config(tmp_config_file)
            net_sections_found = 0
            for section in config_obj.sections():
                if section in config_tup.networks._fields:
                    net_sections_found += 1
                    # check network specific sections
                    net_config_tup = getattr(config_tup.networks, section)
                    self.assertTrue(net_config_tup._fields == NETWORKS[section].Config._fields)
            self.assertTrue(len(config_tup.networks) == net_sections_found)

    def test_invalid_network_option_raises(self):
        test_learncurve_config = os.path.join(TEST_CONFIGS_PATH,
                                              'test_learncurve_config.ini')
        tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_learncurve_config)
        config = ConfigParser()
        config.read(tmp_config_file)
        config['VakTestNet']['bungalow'] = '12'
        with open(tmp_config_file, 'w') as rewrite:
            config.write(rewrite)
        with self.assertRaises(ValueError):
            vak.config.parse_config(tmp_config_file)

    def test_both_train_and_predict_raises(self):
        test_learncurve_config = os.path.join(TEST_CONFIGS_PATH,
                                              'test_learncurve_config.ini')
        tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_learncurve_config)
        config = ConfigParser()
        config.read(tmp_config_file)
        config.add_section('PREDICT')
        config['PREDICT']['networks'] = 'VakTestNet'
        config['PREDICT']['checkpoint_dir'] = self.tmp_checkpoint_dir
        config['PREDICT']['dir_to_predict'] = self.tmp_dir_to_predict
        with open(tmp_config_file, 'w') as rewrite:
            config.write(rewrite)
        with self.assertRaises(ValueError):
            vak.config.parse_config(tmp_config_file)


if __name__ == '__main__':
    unittest.main()
