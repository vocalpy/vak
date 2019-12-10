from pathlib import Path
import unittest

import numpy as np
from scipy.io import loadmat
import crowsetta

import vak.io.dataframe
import vak.io.annotation
from vak.io.classes import MetaSpect


HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestMetaSpect(unittest.TestCase):
    def setUp(self):
        # ---- in .mat files -------------------------------
        self.spect_dir_mat = TEST_DATA_DIR.joinpath('mat', 'llb3', 'spect')
        self.spect_list_mat = list(self.spect_dir_mat.glob('*.mat'))
        self.spect_list_mat = [str(path) for path in self.spect_list_mat]

        self.annot_mat = str(TEST_DATA_DIR.joinpath('mat', 'llb3',
                                                    'llb3_annot_subset.mat'))
        self.scribe = crowsetta.Transcriber(voc_format='yarden')
        self.annot_list = self.scribe.to_seq(self.annot_mat)
        self.labelset_mat = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

        self.spect_params = dict(fft_size=512,
                                 step_size=64,
                                 freq_cutoffs=(500, 10000),
                                 thresh=6.25,
                                 transform_type='log_spect')

        # ---- in .npz files, made from .cbin audio files -------------------------------
        self.spect_dir_npz = list(TEST_DATA_DIR.joinpath('vds').glob(
            'spectrograms_generated*')
        )
        self.spect_dir_npz = self.spect_dir_npz[0]
        self.spect_list_npz = list(self.spect_dir_npz.glob('*.spect.npz'))
        self.spect_list_npz = [str(path) for path in self.spect_list_npz]

    def test_MetaSpect_init_mat(self):
        for spect_path in self.spect_list_mat:
            spect_dict = loadmat(spect_path, squeeze_me=True)
            audio_fname = vak.io.dataframe.find_audio_fname(spect_path)
            a_spect = MetaSpect(freq_bins=spect_dict['f'],
                                time_bins=spect_dict['t'],
                                timebin_dur=0.002,
                                spect=spect_dict['s'],
                                audio_path=audio_fname,
                                )
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'spect', 'audio_path']:
                self.assertTrue(hasattr(a_spect, attr))
                if attr in ['freq_bins', 'time_bins', 'spect']:
                    self.assertTrue(type(getattr(a_spect, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(a_spect, attr)) == float)
                elif attr == 'audio_path':
                    self.assertTrue(type(getattr(a_spect, attr)) == str)

    def test_MetaSpect_init_npz(self):
        for spect_path in self.spect_list_npz:
            spect_dict = np.load(spect_path)
            audio_fname = vak.io.dataframe.find_audio_fname(spect_path)
            a_spect = MetaSpect(freq_bins=spect_dict['f'],
                                time_bins=spect_dict['t'],
                                timebin_dur=0.002,
                                spect=spect_dict['s'],
                                audio_path=audio_fname,
                                )
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'spect', 'audio_path']:
                self.assertTrue(hasattr(a_spect, attr))
                if attr in ['freq_bins', 'time_bins', 'spect']:
                    self.assertTrue(type(getattr(a_spect, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(a_spect, attr)) == float)
                elif attr == 'audio_path':
                    self.assertTrue(type(getattr(a_spect, attr)) == str)

    def test_MetaSpect_from_dict_mat(self):
        for spect_path in self.spect_list_mat:
            spect_dict = loadmat(spect_path, squeeze_me=True)
            metaspect = MetaSpect.from_dict(spect_file_dict=spect_dict,
                                            freqbins_key='f',
                                            timebins_key='t',
                                            spect_key='s',
                                            audio_path_key='audio_path',
                                            timebin_dur=None,
                                            n_decimals_trunc=3)
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'spect', 'audio_path']:
                self.assertTrue(hasattr(metaspect, attr))
                if attr in ['freq_bins', 'time_bins', 'spect']:
                    self.assertTrue(type(getattr(metaspect, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(metaspect, attr)) in (float, np.float16, np.float32, np.float64))
                elif attr == 'audio_path':
                    self.assertTrue(getattr(metaspect, attr) is None)

    def test_MetaSpect_from_dict_npz(self):
        for spect_path in self.spect_list_npz:
            spect_dict = np.load(spect_path)
            audio_fname = vak.io.dataframe.find_audio_fname(spect_path)
            metaspect = MetaSpect.from_dict(spect_file_dict=spect_dict,
                                            freqbins_key='f',
                                            timebins_key='t',
                                            spect_key='s',
                                            audio_path_key='audio_path',
                                            timebin_dur=None,
                                            n_decimals_trunc=3)
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'spect', 'audio_path']:
                self.assertTrue(hasattr(metaspect, attr))
                if attr in ['freq_bins', 'time_bins', 'spect']:
                    self.assertTrue(type(getattr(metaspect, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(metaspect, attr)) in (float, np.float16, np.float32, np.float64))
                elif attr == 'audio_path':
                    self.assertTrue(type(getattr(metaspect, attr)) is str)


if __name__ == '__main__':
    unittest.main()
