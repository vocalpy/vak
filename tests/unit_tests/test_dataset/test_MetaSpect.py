import os
import unittest
from glob import glob

import numpy as np
from scipy.io import loadmat
import crowsetta

import vak.dataset.spect
import vak.dataset.annotation
from vak.dataset.classes import MetaSpect


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestMetaSpect(unittest.TestCase):
    def setUp(self):
        self.spect_dir_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        self.spect_list_mat = glob(os.path.join(self.spect_dir_mat, '*.mat'))

        self.annot_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')
        self.scribe = crowsetta.Transcriber(voc_format='yarden')
        self.annot_list = self.scribe.to_seq(self.annot_mat)
        self.labelset_mat = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

        self.spect_params = dict(fft_size=512,
                                 step_size=64,
                                 freq_cutoffs=(500, 10000),
                                 thresh=6.25,
                                 transform_type='log_spect')

        # ---- cbins -------------------------------
        self.audio_dir_cbin = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        self.audio_files_cbin = glob(os.path.join(self.audio_dir_cbin, '*.cbin'))

        self.annot_files_cbin = vak.dataset.annotation.files_from_dir(annot_dir=self.audio_dir_cbin,
                                                                      annot_format='notmat')
        scribe_cbin = crowsetta.Transcriber(voc_format='notmat')
        self.annot_list_cbin = scribe_cbin.to_seq(file=self.annot_files_cbin)

        self.labelset_cbin = list('iabcdefghjk')

    def test_MetaSpect_init(self):
        for spect_path in self.spect_list_mat:
            spect_dict = loadmat(spect_path, squeeze_me=True)
            a_spect = MetaSpect(freq_bins=spect_dict['f'],
                                time_bins=spect_dict['t'],
                                timebin_dur=0.002,
                                spect=spect_dict['s']
                                )
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'spect']:
                self.assertTrue(hasattr(a_spect, attr))
                if attr in ['freq_bins', 'time_bins', 'spect']:
                    self.assertTrue(type(getattr(a_spect, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(a_spect, attr)) == float)

    def test_MetaSpect_from_dict(self):
        for spect_path in self.spect_list_mat:
            spect_dict = loadmat(spect_path, squeeze_me=True)
            metaspect = MetaSpect.from_dict(spect_file_dict=spect_dict,
                                            freqbins_key='f',
                                            timebins_key='t',
                                            spect_key='s',
                                            timebin_dur=None,
                                            n_decimals_trunc=3)
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'spect']:
                self.assertTrue(hasattr(metaspect, attr))
                if attr in ['freq_bins', 'time_bins', 'spect']:
                    self.assertTrue(type(getattr(metaspect, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(metaspect, attr)) in (float, np.float16, np.float32, np.float64))


if __name__ == '__main__':
    unittest.main()
