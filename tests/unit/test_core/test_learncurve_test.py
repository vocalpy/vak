"""tests for vak.core.learncurve.test module"""
from configparser import ConfigParser
from glob import glob
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import vak.core.learncurve.test
from vak.core.learncurve import LEARN_CURVE_DIR_STEM

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
TEST_CONFIGS_DIR = TEST_DATA_DIR.joinpath('configs')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestLearncurveTest(unittest.TestCase):
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

        results_dir = glob(os.path.join(TEST_DATA_DIR,
                                        'results',
                                        f'{LEARN_CURVE_DIR_STEM}*'))[0]
        config['LEARNCURVE']['results_dir_made_by_main_script'] = results_dir
        with open(self.tmp_config_path, 'w') as fp:
            config.write(fp)

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        shutil.rmtree(self.tmp_config_dir)

    def _check_learncurve_test_output(self):
        test_dirname = os.path.join(self.tmp_output_dir, 'test')
        self.assertTrue(os.path.isdir(test_dirname))
        test_dir_list = os.listdir(test_dirname)
        self.assertTrue('test_err' in test_dir_list)
        self.assertTrue('train_err' in test_dir_list)
        self.assertTrue('Y_pred_test_all' in test_dir_list)
        self.assertTrue('Y_pred_train_all' in test_dir_list)
        self.assertTrue('y_preds_and_err_for_train_and_test' in test_dir_list)

        return True

    def test_learncurve_test(self):
        config = vak.config.parse.parse_config(self.tmp_config_path)
        vak.core.learncurve.test(results_dirname=config.learncurve.results_dirname,
                                 test_vds_path=config.learncurve.test_vds_path,
                                 train_vds_path=config.learncurve.train_vds_path,
                                 networks=config.networks,
                                 train_set_durs=config.learncurve.train_set_durs,
                                 num_replicates=config.learncurve.num_replicates,
                                 output_dir=self.tmp_output_dir,
                                 normalize_spectrograms=config.learncurve.normalize_spectrograms,
                                 save_transformed_data=config.learncurve.save_transformed_data)
        self.assertTrue(self._check_learncurve_test_output())


if __name__ == '__main__':
    unittest.main()
