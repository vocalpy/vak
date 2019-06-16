"""tests for vak.cli.predict module"""
import os
import tempfile
import shutil
import unittest
from configparser import ConfigParser
from glob import glob

import vak.utils
import vak.cli.predict
from vak.core.learncurve import LEARN_CURVE_DIR_STEM

HERE = os.path.dirname(__file__)
TEST_CONFIGS_PATH = os.path.join(HERE, '..', 'test_data', 'configs')
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')


def copydir(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        shutil.copy2(s, d)


@unittest.skip('need to fix predict')
class TestPredict(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()
        a_config = os.path.join(TEST_DATA_DIR, 'configs', 'test_predict_config.ini')
        self.tmp_config_path = os.path.join(TEST_DATA_DIR, 'configs', 'tmp_config.ini')
        shutil.copy(a_config, self.tmp_config_path)

        # copy some data to predict to a temporary dir
        self.tmp_dir_to_predict = tempfile.mkdtemp()
        self.tmp_dir_to_predict = os.path.join(self.tmp_dir_to_predict, '032312')
        os.makedirs(self.tmp_dir_to_predict)
        src = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        copydir(src=src, dst=self.tmp_dir_to_predict)

        a_results_dir = glob(os.path.join(TEST_DATA_DIR,
                                          'results',
                                          f'{LEARN_CURVE_DIR_STEM}*'))[0]
        a_training_records_dir = glob(os.path.join(a_results_dir,
                                                   'train',
                                                   'records_for_training_set*'))[0]
        checkpoint_path = os.path.join(a_training_records_dir, 'TweetyNet', 'checkpoints')
        spect_scaler = glob(os.path.join(a_training_records_dir, 'spect_scaler_*'))[0]

        # rewrite config so it points to data for testing + temporary output dirs
        config = ConfigParser()
        config.read(self.tmp_config_path)
        config['DATA']['data_dir'] = TEST_DATA_DIR
        config['PREDICT']['checkpoint_path'] = checkpoint_path
        config['PREDICT']['dir_to_predict'] = self.tmp_dir_to_predict
        config['PREDICT']['spect_scaler_path'] = spect_scaler
        with open(self.tmp_config_path, 'w') as fp:
            config.write(fp)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir_to_predict)
        os.remove(self.tmp_config_path)

    def test_predict_func(self):
        # make sure predict runs without crashing.
        config = vak.config.parse.parse_config(self.tmp_config_path)
        vak.cli.predict(checkpoint_path=config.predict.checkpoint_path,
                        networks=config.networks,
                        labels_mapping_path=config.predict.labels_mapping_path,
                        spect_params=config.spect_params,
                        dir_to_predict=config.predict.dir_to_predict,
                        spect_scaler_path=config.predict.spect_scaler_path)


if __name__ == '__main__':
    unittest.main()
