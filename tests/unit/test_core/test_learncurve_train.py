"""tests for vak.core.learncurve.test module"""
import os
from pathlib import Path
import tempfile
import shutil
import unittest
from datetime import datetime
from configparser import ConfigParser

import vak.core.learncurve.train
import vak.config
from vak.core.learncurve import LEARN_CURVE_DIR_STEM

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
TEST_CONFIGS_DIR = TEST_DATA_DIR.joinpath('configs')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestLearncurveTrain(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()

        test_learncurve_config = TEST_CONFIGS_DIR.joinpath('test_learncurve_config.ini')
        # Now we want a copy (of the changed version) to use for tests
        # since this is what the test data was made with
        self.tmp_config_dir = tempfile.mkdtemp()
        self.tmp_config_path = Path(self.tmp_config_dir).joinpath('tmp_test_learncurve_config.ini')
        shutil.copy(test_learncurve_config, self.tmp_config_path)

        # rewrite config so it points to data for testing + temporary output dirs
        config = ConfigParser()
        config.read(self.tmp_config_path)
        test_data_vds_path = list(TEST_DATA_DIR.glob('vds'))[0]
        for stem in ['train', 'test', 'val']:
            vds_path = list(test_data_vds_path.glob(f'*.{stem}.vds.json'))
            self.assertTrue(len(vds_path) == 1)
            vds_path = vds_path[0]
            config['LEARNCURVE'][f'{stem}_vds_path'] = str(vds_path)

        config['PREP']['output_dir'] = str(self.tmp_output_dir)
        config['PREP']['data_dir'] = str(TEST_DATA_DIR.joinpath('cbins', 'gy6or6', '032312'))
        config['LEARNCURVE']['root_results_dir'] = str(self.tmp_output_dir)
        with open(self.tmp_config_path, 'w') as fp:
            config.write(fp)

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        shutil.rmtree(self.tmp_config_dir)

    def _check_learncurve_train_output(self, learncurve_config, nets_config, results_dirname):
        train_dirname = os.path.join(results_dirname, 'train')
        train_dir_list = os.listdir(train_dirname)
        records_dirs = [item for item in train_dir_list if 'records' in item]
        self.assertTrue(
            len(records_dirs) == len(learncurve_config.train_set_durs) * learncurve_config.num_replicates
        )

        for record_dir in records_dirs:
            records_path = os.path.join(train_dirname, record_dir)
            records_dir_list = os.listdir(records_path)

            self.assertTrue('train_inds' in records_dir_list)

            for net_name in nets_config.keys():
                self.assertTrue(
                    # make everything lowercase
                    net_name.lower() in [item.lower() for item in records_dir_list]
                )

            if learncurve_config.val_vds_path:
                self.assertTrue('val_errs' in records_dir_list)

            if learncurve_config.save_transformed_data:
                self.assertTrue('X_train' in records_dir_list)
                self.assertTrue('Y_train' in records_dir_list)
                if learncurve_config.val_vds_path:
                    self.assertTrue('X_val' in records_dir_list)
                    self.assertTrue('Y_val' in records_dir_list)
                self.assertTrue('scaled_spects' in records_dir_list)
                self.assertTrue('scaled_reshaped_spects' in records_dir_list)

        return True

    def test_learncurve_train(self):
        config_file = self.tmp_config_path
        config_obj = ConfigParser()
        config_obj.read(config_file)
        learncurve_config = vak.config.parse_learncurve_config(config_obj, config_file)
        nets_config = vak.config.parse._get_nets_config(config_obj, learncurve_config.networks)
        prep_config = vak.config.parse_prep_config(config_obj, config_file)

        timenow = datetime.now().strftime('%y%m%d_%H%M%S')
        results_dirname = os.path.join(learncurve_config.root_results_dir,
                                       f'{LEARN_CURVE_DIR_STEM}{timenow}')
        os.makedirs(results_dirname)

        vak.core.learncurve.train(train_vds_path=learncurve_config.train_vds_path,
                                  total_train_set_duration=prep_config.total_train_set_dur,
                                  train_set_durs=learncurve_config.train_set_durs,
                                  num_replicates=learncurve_config.num_replicates,
                                  num_epochs=learncurve_config.num_epochs,
                                  networks=nets_config,
                                  output_dir=results_dirname,
                                  val_vds_path=learncurve_config.val_vds_path,
                                  val_step=learncurve_config.val_step,
                                  ckpt_step=learncurve_config.ckpt_step,
                                  patience=learncurve_config.patience,
                                  save_only_single_checkpoint_file=learncurve_config.save_only_single_checkpoint_file,
                                  normalize_spectrograms=learncurve_config.normalize_spectrograms,
                                  use_train_subsets_from_previous_run=learncurve_config.use_train_subsets_from_previous_run,
                                  previous_run_path=learncurve_config.previous_run_path,
                                  save_transformed_data=learncurve_config.save_transformed_data)

        self.assertTrue(self._check_learncurve_train_output(
            learncurve_config, nets_config, results_dirname
        ))

    def test_learncurve_train_no_validation(self):
        config_file = self.tmp_config_path
        config_obj = ConfigParser()
        config_obj.read(config_file)
        learncurve_config = vak.config.parse_learncurve_config(config_obj, config_file)
        nets_config = vak.config.parse._get_nets_config(config_obj, learncurve_config.networks)
        prep_config = vak.config.parse_prep_config(config_obj, config_file)

        timenow = datetime.now().strftime('%y%m%d_%H%M%S')
        results_dirname = os.path.join(learncurve_config.root_results_dir,
                                       f'{LEARN_CURVE_DIR_STEM}{timenow}')

        os.makedirs(results_dirname)

        vak.core.learncurve.train(train_vds_path=learncurve_config.train_vds_path,
                                  total_train_set_duration=prep_config.total_train_set_dur,
                                  train_set_durs=learncurve_config.train_set_durs,
                                  num_replicates=learncurve_config.num_replicates,
                                  num_epochs=learncurve_config.num_epochs,
                                  networks=nets_config,
                                  output_dir=results_dirname,
                                  val_vds_path=None,
                                  val_step=None,
                                  ckpt_step=learncurve_config.ckpt_step,
                                  patience=learncurve_config.patience,
                                  save_only_single_checkpoint_file=learncurve_config.save_only_single_checkpoint_file,
                                  normalize_spectrograms=learncurve_config.normalize_spectrograms,
                                  use_train_subsets_from_previous_run=learncurve_config.use_train_subsets_from_previous_run,
                                  previous_run_path=learncurve_config.previous_run_path,
                                  save_transformed_data=learncurve_config.save_transformed_data)

        self.assertTrue(self._check_learncurve_train_output(
            learncurve_config, nets_config, results_dirname
        ))


if __name__ == '__main__':
    unittest.main()
