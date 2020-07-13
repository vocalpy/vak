from pathlib import Path
import unittest

import numpy as np

import vak.files.files
import vak.files.spect
import vak.timebins

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')


class TestGeneral(unittest.TestCase):
    def setUp(self):
        self.mat_spect_path = TEST_DATA_DIR.joinpath('mat', 'llb3', 'spect')
        self.cbin_path = TEST_DATA_DIR.joinpath('cbins', 'gy6or6', '032312')

    def test_find_fname(self):
        fname = 'llb3_0003_2018_04_23_14_18_54.wav.mat'
        ext = 'wav'
        out = vak.files.files.find_fname(fname, ext)
        self.assertTrue(
            out == 'llb3_0003_2018_04_23_14_18_54.wav'
        )

    def test_files_from_dir(self):
        # check works for .mat files
        mat_files = self.mat_spect_path.glob('*.mat')
        mat_files = [str(path) for path in mat_files]
        files = vak.files.files.from_dir(self.mat_spect_path, 'mat')
        self.assertTrue(
            sorted(mat_files) == sorted(files)
        )

        # no reason logically why it shouldn't also work for other file types,
        # but just in case ... we check for an audio file type too
        cbin_files = self.cbin_path.glob('*.cbin')
        cbin_files = [str(path) for path in cbin_files]
        files = vak.files.files.from_dir(self.cbin_path, 'cbin')
        self.assertTrue(
            sorted(cbin_files) == sorted(files)
        )

    def test_timebin_dur_from_vecs(self):
        timebin_dur = 0.001
        time_bins = np.linspace(0., 5., num=int(5 / timebin_dur))
        computed = vak.timebins.timebin_dur_from_vec(time_bins=time_bins, n_decimals_trunc=3)
        self.assertTrue(timebin_dur == computed)


if __name__ == '__main__':
    unittest.main()
