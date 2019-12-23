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
import vak.models
import vak.utils.data
import vak.utils.spect

HERE = os.path.dirname(__file__)
TEST_CONFIGS_PATH = os.path.join(HERE, '..', '..', 'test_data', 'configs')


class TestParseConfig(unittest.TestCase):
    def setUp(self):
        # for data section of config
        self.tmp_data_dir = tempfile.mkdtemp()
        self.tmp_data_output_dir = tempfile.mkdtemp()

        # for train section of config
        _, self.tmp_train_vds_path = tempfile.mkstemp()
        _, self.tmp_val_vds_path = tempfile.mkstemp()
        _, self.tmp_test_vds_path = tempfile.mkstemp()
        _, self.tmp_predict_vds_path = tempfile.mkstemp()

        # for output section of config
        self.tmp_root_dir = tempfile.mkdtemp()
        # have to define this before we refer to it below when "mocking out" objects for predict
        self.tmp_results_dir = tempfile.mkdtemp(dir=self.tmp_root_dir)

        # for predict section of config
        self.tmp_checkpoint_dir = tempfile.mkdtemp()
        self.tmp_dir_to_predict = tempfile.mkdtemp()
        a_spect_scaler = vak.utils.spect.SpectScaler()
        a_spect_scaler.fit(np.random.normal(size=(1000, 513)))
        tmp_spect_scaler_path = os.path.join(self.tmp_results_dir, 'spect_scaler')
        joblib.dump(value=a_spect_scaler, filename=tmp_spect_scaler_path)
        self.tmp_spect_scaler_path = tmp_spect_scaler_path

        # for holding saved temporary config.ini file
        self.tmp_config_dir = tempfile.mkdtemp()

        self.section_to_attr_map = {
            'PREP': 'prep',
            'SPECTROGRAM': 'spect_params',
            'TRAIN': 'train',
            'LEARNCURVE': 'learncurve',
            'PREDICT': 'predict',
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_root_dir)
        shutil.rmtree(self.tmp_data_dir)
        shutil.rmtree(self.tmp_data_output_dir)

        os.remove(self.tmp_train_vds_path)
        os.remove(self.tmp_val_vds_path)
        os.remove(self.tmp_test_vds_path)

    def _add_dirs_to_config_and_save_as_tmp(self, config_file):
        """helper functions called by unit tests to add directories
        that actually exist to avoid spurious NotADirectory errors"""
        config = ConfigParser()
        config.read(config_file)

        if config.has_section('PREP'):
            config['PREP']['data_dir'] = self.tmp_data_dir
            config['PREP']['output_dir'] = self.tmp_data_output_dir

        if config.has_section('TRAIN'):
            config['TRAIN']['train_vds_path'] = self.tmp_train_vds_path
            config['TRAIN']['val_vds_path'] = self.tmp_val_vds_path
            config['TRAIN']['root_results_dir'] = self.tmp_root_dir
            config['TRAIN']['results_dir_made_by_main_script'] = self.tmp_results_dir

        if config.has_section('LEARNCURVE'):
            config['LEARNCURVE']['train_vds_path'] = self.tmp_train_vds_path
            config['LEARNCURVE']['val_vds_path'] = self.tmp_val_vds_path
            config['LEARNCURVE']['test_vds_path'] = self.tmp_test_vds_path
            config['LEARNCURVE']['root_results_dir'] = self.tmp_root_dir
            config['LEARNCURVE']['results_dir_made_by_main_script'] = self.tmp_results_dir

        if config.has_section('PREDICT'):
            config['PREDICT']['checkpoint_path'] = self.tmp_checkpoint_dir
            config['PREDICT']['train_vds_path'] = self.tmp_train_vds_path
            config['PREDICT']['predict_vds_path'] = self.tmp_predict_vds_path
            config['PREDICT']['spect_scaler_path'] = self.tmp_spect_scaler_path

        file_obj = tempfile.NamedTemporaryFile(prefix='config', suffix='.ini', mode='w',
                                               dir=self.tmp_config_dir, delete=False)
        with file_obj as config_file_out:
            config.write(config_file_out)
        return os.path.abspath(file_obj.name)

    def test_parse_config_returns_instance_of_config(self):
        test_learncurve_config = os.path.join(TEST_CONFIGS_PATH,
                                              'test_learncurve_config.ini')
        tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_learncurve_config)
        config_obj = vak.config.parse_config(tmp_config_file)
        self.assertTrue(type(config_obj) == vak.config.parse.Config)

    def test_defined_sections_not_None(self):
        test_configs = glob(os.path.join(TEST_CONFIGS_PATH,
                                         'test_*_config.ini'))
        for test_config in test_configs:
            tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_config)
            config = ConfigParser()
            config.read(tmp_config_file)
            config_obj = vak.config.parse_config(tmp_config_file)
            for section in config.sections():
                if section in self.section_to_attr_map:
                    # check sections that any config.ini file can have, non-network specific
                    attr_name = self.section_to_attr_map[section]
                    self.assertTrue(getattr(config_obj, attr_name) is not None)
                elif section.lower() in config_obj.networks:
                    # check network specific sections
                    self.assertTrue(getattr(config_obj.networks, section) is not None)

    def test_network_sections_match_config(self):
        test_configs = glob(os.path.join(TEST_CONFIGS_PATH,
                                         'test_*_config.ini'))
        NETWORKS = vak.models._load()
        available_net_names = [net_name for net_name in NETWORKS.keys()]
        for test_config in test_configs:
            tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_config)
            config = ConfigParser()
            config.read(tmp_config_file)
            config_obj = vak.config.parse_config(tmp_config_file)
            for section in config.sections():
                if section in available_net_names:
                    net_name_to_check = section
                    self.assertTrue(net_name_to_check in config_obj.networks)
                    # check network specific sections
                    net_config = config_obj.networks[net_name_to_check]
                    self.assertTrue(field in net_config for field in NETWORKS[section].Config._fields)

    def test_invalid_network_option_raises(self):
        test_learncurve_config = os.path.join(TEST_CONFIGS_PATH,
                                              'test_learncurve_config.ini')
        tmp_config_file = self._add_dirs_to_config_and_save_as_tmp(test_learncurve_config)
        config = ConfigParser()
        config.read(tmp_config_file)
        config['TweetyNet']['bungalow'] = '12'
        with open(tmp_config_file, 'w') as rewrite:
            config.write(rewrite)
        with self.assertRaises(ValueError):
            vak.config.parse_config(tmp_config_file)

    def test_invalid_section_raises(self):
        invalid_section_config = os.path.join(
            TEST_CONFIGS_PATH, 'invalid_section_config.ini'
        )
        with self.assertRaises(ValueError):
            vak.config.parse_config(invalid_section_config)

    def test_invalid_option_raises(self):
        invalid_option_config = os.path.join(
            TEST_CONFIGS_PATH, 'invalid_option_config.ini'
        )
        with self.assertRaises(ValueError):
            vak.config.parse_config(invalid_option_config)

    def test_train_and_learncurve_defined_raises(self):
        # should throw a ValueError
        # because both TRAIN and LEARNCURVE sections are defined
        invalid_config = os.path.join(
            TEST_CONFIGS_PATH, 'invalid_train_and_learncurve_config.ini'
        )
        with self.assertRaises(ValueError):
            vak.config.parse_config(invalid_config)


if __name__ == '__main__':
    unittest.main()
