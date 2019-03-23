"""tests for vak.cli.train module"""
import os
import tempfile
import shutil
from glob import glob
import unittest
from configparser import ConfigParser

import vak.cli.train

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()
        # Makefile copies Makefile_config to a tmp version (that gets changed by make_data
        # and other functions)
        tmp_makefile_config = os.path.join(SETUP_SCRIPTS_DIR, 'tmp_Makefile_config.ini')
        # Now we want a copy (of the changed version) to use for tests
        # since this is what the test data was made with
        self.tmp_config_dir = tempfile.mkdtemp()
        self.tmp_config_path = os.path.join(self.tmp_config_dir, 'tmp_config.ini')
        shutil.copy(tmp_makefile_config, self.tmp_config_path)

        # rewrite config so it points to data for testing + temporary output dirs
        config = ConfigParser()
        config.read(self.tmp_config_path)
        test_data_spects_path = glob(os.path.join(TEST_DATA_DIR,
                                                  'spects',
                                                  'spectrograms_*'))[0]
        config['TRAIN']['train_data_path'] = os.path.join(test_data_spects_path, 'train_data_dict')
        config['TRAIN']['val_data_path'] = os.path.join(test_data_spects_path, 'val_data_dict')
        config['TRAIN']['test_data_path'] = os.path.join(test_data_spects_path, 'test_data_dict')
        config['DATA']['output_dir'] = self.tmp_output_dir
        config['DATA']['data_dir'] = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        config['OUTPUT']['root_results_dir'] = self.tmp_output_dir
        with open(self.tmp_config_path, 'w') as fp:
            config.write(fp)

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        shutil.rmtree(self.tmp_config_dir)

    def test_train_func(self):
        # make sure train runs without crashing.
        config = vak.config.parse.parse_config(self.tmp_config_path)
        vak.cli.train(train_data_dict_path=config.train.train_data_dict_path,
                      val_data_dict_path=config.train.val_data_dict_path,
                      spect_params=config.spect_params,
                      networks=config.networks,
                      num_epochs=config.train.num_epochs,
                      config_file=self.tmp_config_path,
                      val_error_step=config.train.val_error_step,
                      checkpoint_step=config.train.checkpoint_step,
                      patience=config.train.patience,
                      save_only_single_checkpoint_file=config.train.save_only_single_checkpoint_file,
                      normalize_spectrograms=config.train.normalize_spectrograms,
                      root_results_dir=config.output.root_results_dir,
                      save_transformed_data=False)


if __name__ == '__main__':
    unittest.main()
