import os
from glob import glob
import unittest

import vak.utils.general


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')


class TestGeneral(unittest.TestCase):
    def setUp(self):
        self.mat_spect_path = os.path.join(TEST_DATA_DIR, 'mat', 'llb11', 'spect')
        self.cbin_path = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')

    def test_files_from_dir(self):
        # check works for .mat files
        mat_files = glob(os.path.join(self.mat_spect_path, 'mat'))
        files = vak.utils.general._files_from_dir(self.mat_spect_path, '*.mat')
        self.assertTrue(
            sorted(mat_files) == sorted(files)
        )

        # no reason logically why it shouldn't also work for other file types, but just in case ...
        # we check for an audio file type too
        cbin_files = glob(os.path.join(self.cbin_path, '*.cbin'))
        files = vak.utils.general._files_from_dir(self.cbin_path, 'cbin')
        self.assertTrue(
            sorted(cbin_files) == sorted(files)
        )


if __name__ == '__main__':
    unittest.main()
