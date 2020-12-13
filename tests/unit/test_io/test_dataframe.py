"""tests for vak.io.dataframe module"""
import os
import tempfile
import shutil
import unittest

import pandas as pd

import vak.io.dataframe
from vak.config.spect_params import SpectParamsConfig

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')


class TestFromFiles(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)

    def test_from_files_with_audio_cbin(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectParamsConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                         transform_type='log_spect')
        annot_format = 'notmat'
        labelset = list('iabcdefghjk')

        vak_df = vak.io.dataframe.from_files(data_dir=data_dir,
                                             labelset=labelset,
                                             annot_format=annot_format,
                                             output_dir=self.tmp_output_dir,
                                             audio_format='cbin',
                                             spect_format=None,
                                             annot_file=None,
                                             spect_params=spect_params)

        self.assertTrue(type(vak_df) == pd.DataFrame)

    def test_from_files_with_audio_cbin_no_annot(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectParamsConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                         transform_type='log_spect')
        annot_format = None
        labelset = None

        vak_df = vak.io.dataframe.from_files(data_dir=data_dir,
                                             annot_format=annot_format,
                                             labelset=labelset,
                                             output_dir=self.tmp_output_dir,
                                             audio_format='cbin',
                                             spect_format=None,
                                             annot_file=None,
                                             spect_params=spect_params)

        self.assertTrue(type(vak_df) == pd.DataFrame)

    def test_from_files_with_audio_cbin_no_labelset(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectParamsConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                         transform_type='log_spect')
        annot_format = 'notmat'
        labelset = None

        vak_df = vak.io.dataframe.from_files(data_dir=data_dir,
                                             annot_format=annot_format,
                                             labelset=labelset,
                                             output_dir=self.tmp_output_dir,
                                             audio_format='cbin',
                                             spect_format=None,
                                             annot_file=None,
                                             spect_params=spect_params)

        self.assertTrue(type(vak_df) == pd.DataFrame)

    def test_from_files_with_spect_mat(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        annot_file = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')

        labelset = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}
        annot_format = 'yarden'

        vak_df = vak.io.dataframe.from_files(data_dir=data_dir,
                                             labelset=labelset,
                                             annot_format=annot_format,
                                             output_dir=self.tmp_output_dir,
                                             audio_format=None,
                                             spect_format='mat',
                                             annot_file=annot_file,
                                             spect_params=None)

        self.assertTrue(type(vak_df) == pd.DataFrame)

    def test_from_files_with_spect_mat_no_annot(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        annot_file = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')
        labelset = None
        annot_format = None

        vak_df = vak.io.dataframe.from_files(data_dir=data_dir,
                                             labelset=labelset,
                                             annot_format=annot_format,
                                             output_dir=self.tmp_output_dir,
                                             audio_format=None,
                                             spect_format='mat',
                                             annot_file=annot_file,
                                             spect_params=None)

        self.assertTrue(type(vak_df) == pd.DataFrame)

    def test_from_files_with_spect_mat_no_labelset(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        annot_file = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')

        labelset = None
        annot_format = 'yarden'

        vak_df = vak.io.dataframe.from_files(data_dir=data_dir,
                                             labelset=labelset,
                                             annot_format=annot_format,
                                             output_dir=self.tmp_output_dir,
                                             audio_format=None,
                                             spect_format='mat',
                                             annot_file=annot_file,
                                             spect_params=None)

        self.assertTrue(type(vak_df) == pd.DataFrame)


class TestAddSplitCol(unittest.TestCase):
    """class to test vak.io.dataframe.add_split_col function"""
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)

    def test_add_split_col(self):
        # make a df to test on
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectParamsConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                         transform_type='log_spect')
        annot_format = 'notmat'
        labelset = list('iabcdefghjk')

        vak_df = vak.io.dataframe.from_files(data_dir=data_dir,
                                             labelset=labelset,
                                             annot_format=annot_format,
                                             output_dir=self.tmp_output_dir,
                                             audio_format='cbin',
                                             spect_format=None,
                                             annot_file=None,
                                             spect_params=spect_params)

        self.assertTrue(
            'split' not in vak_df.columns
        )
        vak_df = vak.io.dataframe.add_split_col(vak_df, split='train')
        self.assertTrue(
            'split' in vak_df.columns
        )
        self.assertTrue(
            vak_df['split'].unique().item() == 'train'
        )


if __name__ == '__main__':
    unittest.main()
