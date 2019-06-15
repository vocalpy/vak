"""tests for vak.cli.learncurve module"""
from configparser import ConfigParser
from glob import glob
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import vak.cli.summary
from vak.core.learncurve import LEARN_CURVE_DIR_STEM

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestSummary(unittest.TestCase):
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

        config['DATA']['output_dir'] = str(self.tmp_output_dir)
        config['DATA']['data_dir'] = str(TEST_DATA_DIR.joinpath('cbins', 'gy6or6', '032312'))
        config['OUTPUT']['root_results_dir'] = str(self.tmp_output_dir)
        with open(self.tmp_config_path, 'w') as fp:
            config.write(fp)

        results_dir = glob(os.path.join(TEST_DATA_DIR,
                                        'results',
                                        f'{LEARN_CURVE_DIR_STEM}*'))[0]
        config['OUTPUT']['results_dir_made_by_main_script'] = results_dir
        with open(self.tmp_config_path, 'w') as fp:
            config.write(fp)

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        shutil.rmtree(self.tmp_config_dir)

    def test_summary_func(self):
        # make sure cli.summary runs without crashing.
        config = vak.config.parse.parse_config(self.tmp_config_path)
        vak.cli.summary(results_dirname=config.output.results_dirname,
                        test_vds_path=config.train.test_vds_path,
                        train_vds_path=config.train.train_vds_path,
                        networks=config.networks,
                        train_set_durs=config.train.train_set_durs,
                        num_replicates=config.train.num_replicates,
                        normalize_spectrograms=config.train.normalize_spectrograms,
                        save_transformed_data=config.data.save_transformed_data)


if __name__ == '__main__':
    unittest.main()
