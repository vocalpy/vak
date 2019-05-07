import os
import unittest
from glob import glob

import crowsetta

import vak.dataset.array


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestArray(unittest.TestCase):
    def setUp(self):
        self.array_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb11', 'spect')
        self.array_list = glob(os.path.join(self.array_dir, '*.mat'))
        self.array_format = 'mat'

        self.annot_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb11', 'llb11_annot_subset.mat')
        self.scribe = crowsetta.Transcriber(voc_format='yarden')
        self.annot_list = self.scribe.to_seq(self.annot_mat)

    def test_array_dir_annot(self):
        vocal_dataset = vak.dataset.array.from_arr_files(self.array_format,
                                                         array_dir=self.array_dir,
                                                         annot_list=self.annot_list)
        array_list_basenames = [os.path.basename(arr_path) for arr_path in self.array_list]
        spect_files = [os.path.basename(voc.spect_file)
                       for voc in vocal_dataset.voc_list]

        self.assertTrue(
            all([spect_file in array_list_basenames for spect_file in spect_files])
        )


if __name__ == '__main__':
    unittest.main()
