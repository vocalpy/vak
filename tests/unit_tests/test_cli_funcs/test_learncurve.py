"""tests for vak.cli.learncurve module"""
import os
from pathlib import Path
import tempfile
import shutil
import unittest
from datetime import datetime
from configparser import ConfigParser

import vak.cli.learncurve
import vak.config
from vak.core.learncurve import LEARN_CURVE_DIR_STEM

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestLearncurve(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()
        # Makefile copies Makefile_config to a tmp version (that gets changed by make_data
        # and other functions)
        tmp_makefile_config = SETUP_SCRIPTS_DIR.joinpath('tmp_Makefile_config.ini')
        # Now we want a copy (of the changed version) to use for tests
        # since this is what the test data was made with
        self.tmp_config_dir = tempfile.mkdtemp()
        self.tmp_config_path = Path(self.tmp_config_dir).joinpath('tmp_config.ini')
        shutil.copy(tmp_makefile_config, self.tmp_config_path)

        # rewrite config so it points to data for testing + temporary output dirs
        config = ConfigParser()
        config.read(self.tmp_config_path)
        test_data_vds_path = list(TEST_DATA_DIR.glob('vds'))[0]
        for stem in ['train', 'test', 'val']:
            vds_path = list(test_data_vds_path.glob(f'*.{stem}.vds.json'))
            self.assertTrue(len(vds_path) == 1)
            vds_path = vds_path[0]
            config['TRAIN'][f'{stem}_vds_path'] = str(vds_path)

        config['PREP']['output_dir'] = str(self.tmp_output_dir)
        config['PREP']['data_dir'] = str(TEST_DATA_DIR.joinpath('cbins', 'gy6or6', '032312'))
        config['OUTPUT']['root_results_dir'] = str(self.tmp_output_dir)
        with open(self.tmp_config_path, 'w') as fp:
            config.write(fp)

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        shutil.rmtree(self.tmp_config_dir)

    def _check_learncurve_output(self, output_config, train_config, nets_config, prep_config,
                                 time_before, time_after):
        output_dir_after = os.listdir(output_config.root_results_dir)
        self.assertTrue(len(output_dir_after) == 1)

        results_dir = output_dir_after[0]
        self.assertTrue(LEARN_CURVE_DIR_STEM in results_dir)

        time_str_results_dir = results_dir.replace(LEARN_CURVE_DIR_STEM, '')  # to get just datestr
        time_results_dir = datetime.strptime(time_str_results_dir, '%y%m%d_%H%M%S')
        self.assertTrue(time_before <= time_results_dir <= time_after)

        # -------- test output of learncurve.train --------------------------------------------------------------------
        train_dirname = os.path.join(output_config.root_results_dir, results_dir, 'train')
        train_dir_list = os.listdir(train_dirname)
        records_dirs = [item for item in train_dir_list if 'records' in item]
        self.assertTrue(
            len(records_dirs) == len(train_config.train_set_durs) * train_config.num_replicates
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

            if train_config.val_vds_path:
                self.assertTrue('val_errs' in records_dir_list)

            if train_config.save_transformed_data:
                self.assertTrue('X_train' in records_dir_list)
                self.assertTrue('Y_train' in records_dir_list)
                if train_config.val_vds_path:
                    self.assertTrue('X_val' in records_dir_list)
                    self.assertTrue('Y_val' in records_dir_list)
                self.assertTrue('scaled_spects' in records_dir_list)
                self.assertTrue('scaled_reshaped_spects' in records_dir_list)

        # -------- test output of learncurve.test ---------------------------------------------------------------------
        test_dirname = os.path.join(output_config.root_results_dir, results_dir, 'test')
        self.assertTrue(os.path.isdir(test_dirname))
        test_dir_list = os.listdir(test_dirname)
        self.assertTrue('test_err' in test_dir_list)
        self.assertTrue('train_err' in test_dir_list)
        self.assertTrue('Y_pred_test_all' in test_dir_list)
        self.assertTrue('Y_pred_train_all' in test_dir_list)
        self.assertTrue('y_preds_and_err_for_train_and_test' in test_dir_list)

        return True

    def test_learncurve_func(self):
        # this kind of repeats what happens in self.setUp, but
        # this is the way cli does it using what user passed in
        # so we repeat that logic here
        config_file = self.tmp_config_path
        config_obj = ConfigParser()
        config_obj.read(config_file)
        train_config = vak.config.parse_train_config(config_obj, config_file)
        nets_config = vak.config.parse._get_nets_config(config_obj, train_config.networks)
        prep_config = vak.config.parse_prep_config(config_obj, config_file)
        output_config = vak.config.parse_output_config(config_obj)

        # want time to make sure results dir generated has correct time;
        # have to drop microseconds from datetime object because we don't include that in
        # the string format that's in the directory name, and if we keep it here then
        # the time recovered from the directory name can be "less than" the time
        # from before starting--i.e. some datetime with microseconds is less than the
        # exact same date time but with some number of microseconds
        time_before = datetime.now().replace(microsecond=0)
        vak.cli.learning_curve(train_vds_path=train_config.train_vds_path,
                               test_vds_path=train_config.test_vds_path,
                               total_train_set_duration=prep_config.total_train_set_dur,
                               train_set_durs=train_config.train_set_durs,
                               num_replicates=train_config.num_replicates,
                               num_epochs=train_config.num_epochs,
                               config_file=config_file,
                               networks=nets_config,
                               val_vds_path=train_config.val_vds_path,
                               val_error_step=train_config.val_error_step,
                               checkpoint_step=train_config.checkpoint_step,
                               patience=train_config.patience,
                               save_only_single_checkpoint_file=train_config.save_only_single_checkpoint_file,
                               normalize_spectrograms=train_config.normalize_spectrograms,
                               use_train_subsets_from_previous_run=train_config.use_train_subsets_from_previous_run,
                               previous_run_path=train_config.previous_run_path,
                               root_results_dir=output_config.root_results_dir,
                               save_transformed_data=train_config.save_transformed_data)
        time_after = datetime.now().replace(microsecond=0)
        self.assertTrue(self._check_learncurve_output(
            output_config, train_config, nets_config, prep_config, time_before, time_after
        ))

    def test_learncurve_no_validation(self):
        # this kind of repeats what happens in self.setUp, but
        # this is the way cli does it using what user passed in
        # so we repeat that logic here
        config_file = self.tmp_config_path
        config_obj = ConfigParser()
        config_obj.read(config_file)
        train_config = vak.config.parse_train_config(config_obj, config_file)
        nets_config = vak.config.parse._get_nets_config(config_obj, train_config.networks)
        prep_config = vak.config.parse_prep_config(config_obj, config_file)
        output_config = vak.config.parse_output_config(config_obj)

        # want time to make sure results dir generated has correct time;
        # have to drop microseconds from datetime object because we don't include that in
        # the string format that's in the directory name, and if we keep it here then
        # the time recovered from the directory name can be "less than" the time
        # from before starting--i.e. some datetime with microseconds is less than the
        # exact same date time but with some number of microseconds
        time_before = datetime.now().replace(microsecond=0)
        vak.cli.learning_curve(train_vds_path=train_config.train_vds_path,
                               test_vds_path=train_config.test_vds_path,
                               total_train_set_duration=prep_config.total_train_set_dur,
                               train_set_durs=train_config.train_set_durs,
                               num_replicates=train_config.num_replicates,
                               num_epochs=train_config.num_epochs,
                               config_file=config_file,
                               networks=nets_config,
                               val_vds_path=None,
                               val_error_step=None,
                               checkpoint_step=train_config.checkpoint_step,
                               patience=train_config.patience,
                               save_only_single_checkpoint_file=train_config.save_only_single_checkpoint_file,
                               normalize_spectrograms=train_config.normalize_spectrograms,
                               use_train_subsets_from_previous_run=train_config.use_train_subsets_from_previous_run,
                               previous_run_path=train_config.previous_run_path,
                               root_results_dir=output_config.root_results_dir,
                               save_transformed_data=train_config.save_transformed_data)
        time_after = datetime.now().replace(microsecond=0)
        self.assertTrue(self._check_learncurve_output(
            output_config, train_config, nets_config, prep_config, time_before, time_after
        ))


if __name__ == '__main__':
    unittest.main()
