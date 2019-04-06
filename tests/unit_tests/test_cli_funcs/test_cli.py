"""tests for vak.cli module"""
import os
import tempfile
import shutil
import unittest
from configparser import ConfigParser
from glob import glob

import vak.utils
import vak.cli.cli

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')
TEST_CONFIGS_PATH = os.path.join(TEST_DATA_DIR, 'configs')


def copydir(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        shutil.copy2(s, d)


class TestCli(unittest.TestCase):
    def setUp(self):
        # copy temporary configs inside TEST_CONFIGS_PATH
        predict_config = os.path.join(TEST_CONFIGS_PATH, 'test_predict_config.ini')
        learncurve_config = os.path.join(TEST_CONFIGS_PATH, 'test_learncurve_config.ini')
        self.tmp_predict_config_path = os.path.join(TEST_CONFIGS_PATH, 'tmp_predict_config.ini')
        self.tmp_learncurve_config_path = os.path.join(TEST_CONFIGS_PATH, 'tmp_learncurve_config.ini')
        shutil.copy(predict_config, self.tmp_predict_config_path)
        shutil.copy(learncurve_config, self.tmp_learncurve_config_path)

        # make temporary otuput dir
        self.tmp_output_dir = tempfile.mkdtemp()

        # copy some data to predict to a temporary dir
        self.tmp_dir_to_predict = tempfile.mkdtemp()
        self.tmp_dir_to_predict = os.path.join(self.tmp_dir_to_predict, '032312')
        os.makedirs(self.tmp_dir_to_predict)
        src = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        copydir(src=src, dst=self.tmp_dir_to_predict)

        a_results_dir = glob(os.path.join(TEST_DATA_DIR, 'results', 'results_*'))[0]
        labels_mapping_path = glob(os.path.join(a_results_dir, 'labels_mapping'))[0]
        a_training_records_dir = glob(os.path.join(a_results_dir,
                                                   'records_for_training_set*')
                                      )[0]
        checkpoint_path = os.path.join(a_training_records_dir, 'TweetyNet', 'checkpoints')
        spect_scaler = glob(os.path.join(a_training_records_dir, 'spect_scaler_*'))[0]

        for tmp_config_path in (self.tmp_predict_config_path,
                                self.tmp_learncurve_config_path):
            # rewrite config so it points to data for testing + temporary output dirs
            config = ConfigParser()
            config.read(tmp_config_path)
            config['DATA']['data_dir'] = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
            config['DATA']['output_dir'] = self.tmp_output_dir

            if config.has_section('PREDICT'):
                config['PREDICT']['checkpoint_path'] = checkpoint_path
                config['PREDICT']['labels_mapping_path'] = labels_mapping_path
                config['PREDICT']['dir_to_predict'] = self.tmp_dir_to_predict
                config['PREDICT']['spect_scaler_path'] = spect_scaler

            if config.has_section('TRAIN'):
                test_data_spects_path = glob(os.path.join(TEST_DATA_DIR,
                                                          'spects',
                                                          'spectrograms_*'))[0]
                config['TRAIN']['train_data_path'] = os.path.join(test_data_spects_path, 'train_data_dict')
                config['TRAIN']['val_data_path'] = os.path.join(test_data_spects_path, 'val_data_dict')
                config['TRAIN']['test_data_path'] = os.path.join(test_data_spects_path, 'test_data_dict')
                config['OUTPUT']['root_results_dir'] = self.tmp_output_dir
                config['OUTPUT']['results_dir_made_by_main_script'] = glob(os.path.join(TEST_DATA_DIR,
                                                                                        'results',
                                                                                        'results_*'))[0]

            with open(tmp_config_path, 'w') as fp:
                config.write(fp)

    def tearDown(self):
        os.remove(self.tmp_predict_config_path)
        os.remove(self.tmp_learncurve_config_path)
        shutil.rmtree(self.tmp_output_dir)
        shutil.rmtree(self.tmp_dir_to_predict)

    def test_prep_command(self):
        command = 'prep'
        config_files = [self.tmp_learncurve_config_path]
        vak.cli.cli(command=command, config_files=config_files)

    def test_train_command(self):
        command = 'train'
        config_files = [self.tmp_learncurve_config_path]
        vak.cli.cli(command=command, config_files=config_files)

    def test_predict_command(self):
        command = 'predict'
        config_files = [self.tmp_predict_config_path]
        vak.cli.cli(command=command, config_files=config_files)

    def test_learncurve_command(self):
        command = 'learncurve'
        config_files = [self.tmp_learncurve_config_path]
        vak.cli.cli(command=command, config_files=config_files)

    def test_summary_command(self):
        command = 'summary'
        config_files = [self.tmp_learncurve_config_path]
        vak.cli.cli(command=command, config_files=config_files)


if __name__ == '__main__':
    unittest.main()
