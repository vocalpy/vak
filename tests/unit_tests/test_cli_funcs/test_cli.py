"""tests for vak.cli module"""
from configparser import ConfigParser
from glob import glob
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import vak.utils
import vak.cli.cli
from vak.core.learncurve import LEARN_CURVE_DIR_STEM

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
TEST_CONFIGS_PATH = TEST_DATA_DIR.joinpath('configs')


def copydir(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        shutil.copy2(s, d)


class TestCli(unittest.TestCase):
    def setUp(self):
        # copy temporary configs inside TEST_CONFIGS_PATH
        predict_config = TEST_CONFIGS_PATH.joinpath('test_predict_config.ini')
        learncurve_config = TEST_CONFIGS_PATH.joinpath('test_learncurve_config.ini')
        self.tmp_config_dir = Path(tempfile.mkdtemp())
        self.tmp_predict_config_path = self.tmp_config_dir.joinpath(
            'tmp_predict_config.ini')
        self.tmp_learncurve_config_path = self.tmp_config_dir.joinpath(
            'tmp_learncurve_config.ini')
        shutil.copy(predict_config, self.tmp_predict_config_path)
        shutil.copy(learncurve_config, self.tmp_learncurve_config_path)

        # make temporary output dir
        self.tmp_output_dir = tempfile.mkdtemp()

        # copy some data to predict to a temporary dir
        self.tmp_dir_to_predict = tempfile.mkdtemp()
        self.tmp_dir_to_predict = os.path.join(self.tmp_dir_to_predict, '032312')
        os.makedirs(self.tmp_dir_to_predict)
        src = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        copydir(src=src, dst=self.tmp_dir_to_predict)

        a_results_dir = TEST_DATA_DIR.joinpath(
            'results'
        ).glob(f'{LEARN_CURVE_DIR_STEM}*')
        a_results_dir = list(a_results_dir)[0]
        a_training_records_dir = glob(os.path.join(a_results_dir,
                                                   'train',
                                                   'records_for_training_set*'))[0]
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
                config['PREDICT']['dir_to_predict'] = self.tmp_dir_to_predict
                config['PREDICT']['spect_scaler_path'] = spect_scaler

            if config.has_section('TRAIN'):
                test_data_vds_path = TEST_DATA_DIR.joinpath('vds')

                train_vds_path = test_data_vds_path.glob('*train.vds.json')
                train_vds_path = str(list(train_vds_path)[0])
                config['TRAIN']['train_vds_path'] = train_vds_path

                val_vds_path = test_data_vds_path.glob('*val.vds.json')
                val_vds_path = str(list(val_vds_path)[0])
                config['TRAIN']['val_vds_path'] = val_vds_path

                test_vds_path = test_data_vds_path.glob('*test.vds.json')
                test_vds_path = str(list(test_vds_path)[0])
                config['TRAIN']['test_vds_path'] = test_vds_path

                config['OUTPUT']['root_results_dir'] = self.tmp_output_dir
                config['OUTPUT']['results_dir_made_by_main_script'] = str(a_results_dir)

            with open(tmp_config_path, 'w') as fp:
                config.write(fp)

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        shutil.rmtree(self.tmp_dir_to_predict)
        shutil.rmtree(self.tmp_config_dir)

    def test_prep_command(self):
        # remove data path options, this function should work without them
        # present in .ini file, and should add them when it runs
        config = ConfigParser()
        config.read(self.tmp_learncurve_config_path)
        config.remove_option('TRAIN', 'train_vds_path')
        config.remove_option('TRAIN', 'val_vds_path')
        config.remove_option('TRAIN', 'test_vds_path')
        with open(self.tmp_learncurve_config_path, 'w') as fp:
            config.write(fp)

        vak.cli.cli(command='prep', config_file=self.tmp_learncurve_config_path)

        # assert that data path options got added
        config = ConfigParser()
        config.read(self.tmp_learncurve_config_path)
        for option in ('train_vds_path', 'val_vds_path', 'test_vds_path'):
            self.assertTrue(config.has_option('TRAIN', option))

    def test_train_command(self):
        vak.cli.cli(command='train', config_file=self.tmp_learncurve_config_path)

    @unittest.skip('need to fix predict')
    def test_predict_command(self):
        vak.cli.cli(command='predict', config_file=self.tmp_predict_config_path)

    def test_learncurve_command(self):
        # remove option that should not be defined yet so it doesn't cause crash
        config = ConfigParser()
        config.read(self.tmp_learncurve_config_path)
        config.remove_option('OUTPUT', 'results_dir_made_by_main_script')
        with open(self.tmp_learncurve_config_path, 'w') as fp:
            config.write(fp)

        vak.cli.cli(command='learncurve', config_file=self.tmp_learncurve_config_path)

    def test_learncurve_with_results_dir_raises(self):
        with self.assertRaises(ValueError):
            vak.cli.cli(command='learncurve', config_file=self.tmp_learncurve_config_path)


if __name__ == '__main__':
    unittest.main()
