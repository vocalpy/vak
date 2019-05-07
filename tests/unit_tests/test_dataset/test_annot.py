import os
import unittest
from glob import glob

import vak.dataset.annot


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestAnnot(unittest.TestCase):
    def test_files_from_dir(self):
        notmat_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        annot_files = vak.dataset.annot.files_from_dir(notmat_dir, annot_format='notmat')

        notmat_files = glob(os.path.join(notmat_dir, '*.not.mat'))
        self.assertTrue(
            sorted(annot_files) == sorted(notmat_files)
        )


if __name__ == '__main__':
    unittest.main()
