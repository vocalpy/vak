"""tests for vak.cli.learncurve module"""
import os
import tempfile
import shutil
from glob import glob
import unittest
from datetime import datetime
from configparser import ConfigParser

import vak.cli.learncurve
import vak.config

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestLearncurve(unittest.TestCase):
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

    def _check_learncurve_output(self, output_config, train_config, nets_config, data_config,
                                 time_before, time_after):
        root_results_after = os.listdir(output_config.root_results_dir)
        self.assertTrue(len(root_results_after) == 1)

        results_dir = root_results_after[0]
        self.assertTrue('results_' in results_dir)

        time_str_results_dir = results_dir.replace('results_', '')  # to get just datestr
        time_results_dir = datetime.strptime(time_str_results_dir, '%y%m%d_%H%M%S')
        self.assertTrue(time_before <= time_results_dir <= time_after)

        results_path = os.path.join(output_config.root_results_dir, results_dir)
        results_dir_list = os.listdir(results_path)
        records_dirs = [item for item in results_dir_list if 'records' in item]
        self.assertTrue(
            len(records_dirs) == len(train_config.train_set_durs) * train_config.num_replicates
        )

        for record_dir in records_dirs:
            records_path = os.path.join(results_path, record_dir)
            records_dir_list = os.listdir(records_path)

            self.assertTrue('train_inds' in records_dir_list)

            for net_name in nets_config.keys():
                self.assertTrue(
                    # make everything lowercase
                    net_name.lower() in [item.lower() for item in records_dir_list]
                )

            if train_config.val_data_dict_path:
                self.assertTrue('val_errs' in records_dir_list)

            if data_config.save_transformed_data:
                self.assertTrue('val_errs' in records_dir_list)
                self.assertTrue('val_errs' in records_dir_list)

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
        spect_params = vak.config.parse_spect_config(config_obj)
        data_config = vak.config.parse_data_config(config_obj, config_file)
        output_config = vak.config.parse_output_config(config_obj)

        # want time to make sure results dir generated has correct time;
        # have to drop microseconds from datetime object because we don't include that in
        # the string format that's in the directory name, and if we keep it here then
        # the time recovered from the directory name can be "less than" the time
        # from before starting--i.e. some datetime with microseconds is less than the
        # exact same date time but with some number of microseconds
        time_before = datetime.now().replace(microsecond=0)
        vak.cli.learncurve(train_data_dict_path=train_config.train_data_dict_path,
                           val_data_dict_path=train_config.val_data_dict_path,
                           spect_params=spect_params,
                           total_train_set_duration=data_config.total_train_set_dur,
                           train_set_durs=train_config.train_set_durs,
                           num_replicates=train_config.num_replicates,
                           num_epochs=train_config.num_epochs,
                           config_file=config_file,
                           networks=nets_config,
                           val_error_step=train_config.val_error_step,
                           checkpoint_step=train_config.checkpoint_step,
                           patience=train_config.patience,
                           save_only_single_checkpoint_file=train_config.save_only_single_checkpoint_file,
                           normalize_spectrograms=train_config.normalize_spectrograms,
                           use_train_subsets_from_previous_run=train_config.use_train_subsets_from_previous_run,
                           previous_run_path=train_config.previous_run_path,
                           root_results_dir=output_config.root_results_dir,
                           save_transformed_data=data_config.save_transformed_data)
        time_after = datetime.now().replace(microsecond=0)
        self.assertTrue(self._check_learncurve_output(
            output_config, train_config, nets_config, data_config, time_before, time_after
        ))


if __name__ == '__main__':
    unittest.main()
