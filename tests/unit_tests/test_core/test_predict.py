"""tests for vak.cli.predict module"""
from configparser import ConfigParser
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import crowsetta
import numpy as np

import vak.core.predict
from vak.core.learncurve import LEARN_CURVE_DIR_STEM
from vak.io import Dataset
import vak.utils

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
TEST_CONFIGS_PATH = TEST_DATA_DIR.joinpath('configs')


def copydir(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        shutil.copy2(s, d)


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()

        a_results_dir = list(
            TEST_DATA_DIR.joinpath('results').glob(
                f'{LEARN_CURVE_DIR_STEM}*'))[0]
        a_training_records_dir = list(
            Path(a_results_dir).joinpath(
                'train').glob('records_for_training_set*'))[0]
        self.checkpoint_path = str(Path(a_training_records_dir).joinpath(
            'TweetyNet', 'checkpoints'))
        spect_scaler = list(
            Path(a_training_records_dir).glob('spect_scaler_*'))[0]
        self.spect_scaler_path = str(spect_scaler)

        # rewrite config so it points to data for testing + temporary output dirs
        # rewrite config so it points to data for testing + temporary output dirs
        a_config = str(TEST_CONFIGS_PATH.joinpath('test_predict_config.ini'))
        config = ConfigParser()
        config.read(a_config)
        config['PREDICT']['checkpoint_path'] = self.checkpoint_path
        config['PREDICT']['spect_scaler_path'] = self.spect_scaler_path

        test_data_vds_path = list(TEST_DATA_DIR.glob('vds'))[0]

        vds_path = list(test_data_vds_path.glob(f'*.train.vds.json'))
        self.assertTrue(len(vds_path) == 1)
        vds_path = vds_path[0]
        self.train_vds_path = str(shutil.copy(vds_path, self.tmp_output_dir))
        config['PREDICT']['train_vds_path'] = self.train_vds_path

        train_vds = Dataset.load(json_fname=vds_path)
        if train_vds.are_spects_loaded() is False:
            train_vds = train_vds.load_spects()
        self.labelmap = train_vds.labelmap
        del train_vds

        vds_path = list(test_data_vds_path.glob(f'*.predict.vds.json'))
        self.assertTrue(len(vds_path) == 1)
        vds_path = vds_path[0]
        self.predict_vds_path = str(shutil.copy(vds_path, self.tmp_output_dir))
        config['PREDICT']['predict_vds_path'] = self.predict_vds_path

        self.config_obj = config

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)

    def test_predict_func(self):
        # make sure we're working with a vds that needs prediction
        predict_vds_before = Dataset.load(self.predict_vds_path)
        predict_vds_before = predict_vds_before.load_spects()
        for voc in predict_vds_before.voc_list:
            self.assertTrue(voc.metaspect.lbl_tb is None)
            self.assertTrue(voc.annot is None)

        predict_config = vak.config.predict.parse_predict_config(self.config_obj)
        networks = vak.config.parse._get_nets_config(self.config_obj,
                                                     predict_config.networks)
        vak.core.predict(predict_vds_path=self.predict_vds_path,
                         labelmap=self.labelmap,
                         checkpoint_path=self.checkpoint_path,
                         networks=networks,
                         spect_scaler_path=self.spect_scaler_path)
        predict_vds_after = Dataset.load(self.predict_vds_path)
        predict_vds_after = predict_vds_after.load_spects()

        for voc in predict_vds_after.voc_list:
            self.assertTrue(type(voc.annot) == crowsetta.Sequence)
            self.assertTrue(type(voc.annot.labels) is np.ndarray)


if __name__ == '__main__':
    unittest.main()
