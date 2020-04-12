"""tests for vak.cli.train module"""
from configparser import ConfigParser
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import vak.cli.train


HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
TEST_CONFIGS_DIR = TEST_DATA_DIR.joinpath('configs')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()

        test_train_config = TEST_CONFIGS_DIR.joinpath('test_train_config.ini')
        self.tmp_config_dir = tempfile.mkdtemp()
        self.tmp_config_path = os.path.join(self.tmp_config_dir, 'tmp_test_train_config.ini')
        shutil.copy(test_train_config, self.tmp_config_path)

        # rewrite config so it points to data for testing + temporary output dirs
        config = ConfigParser()
        config.read(self.tmp_config_path)
        test_data_vds_path = list(TEST_DATA_DIR.glob('vds'))[0]
        for stem in ['train', 'val']:
            vds_path = list(test_data_vds_path.glob(f'*.{stem}.vds.json'))
            self.assertTrue(len(vds_path) == 1)
            vds_path = vds_path[0]
            config['TRAIN'][f'{stem}_vds_path'] = str(vds_path)
        config['PREP']['output_dir'] = self.tmp_output_dir
        config['PREP']['data_dir'] = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        config['TRAIN']['root_results_dir'] = self.tmp_output_dir
        with open(self.tmp_config_path, 'w') as fp:
            config.write(fp)

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        shutil.rmtree(self.tmp_config_dir)

    def test_train_func(self):
        # make sure train runs without crashing.
        config = vak.config.parse.parse_config(self.tmp_config_path)
        vak.cli.train(train_vds_path=config.train.train_vds_path,
                      val_vds_path=config.train.val_vds_path,
                      networks=config.networks,
                      num_epochs=config.train.num_epochs,
                      config_file=self.tmp_config_path,
                      val_step=config.train.val_step,
                      ckpt_step=config.train.ckpt_step,
                      patience=config.train.patience,
                      save_only_single_checkpoint_file=config.train.save_only_single_checkpoint_file,
                      normalize_spectrograms=config.train.normalize_spectrograms,
                      root_results_dir=config.train.root_results_dir,
                      save_transformed_data=False)

    def test_train_func_no_val_set(self):
        # make sure train runs without a validation set
        config = vak.config.parse.parse_config(self.tmp_config_path)
        vak.cli.train(train_vds_path=config.train.train_vds_path,
                      val_vds_path=None,
                      networks=config.networks,
                      num_epochs=config.train.num_epochs,
                      config_file=self.tmp_config_path,
                      val_step=None,
                      ckpt_step=config.train.ckpt_step,
                      patience=config.train.patience,
                      save_only_single_checkpoint_file=config.train.save_only_single_checkpoint_file,
                      normalize_spectrograms=config.train.normalize_spectrograms,
                      root_results_dir=config.train.root_results_dir,
                      save_transformed_data=False)


if __name__ == '__main__':
    unittest.main()
