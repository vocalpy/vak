import os
import unittest
from glob import glob

import numpy as np
from scipy.io import loadmat
import crowsetta

from vak.evfuncs import load_cbin
import vak.io.dataframe
import vak.io.annotation
from vak.io.classes import Vocalization, MetaSpect


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestVocalization(unittest.TestCase):
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

        self.annot_files_cbin = vak.io.annotation.files_from_dir(annot_dir=self.audio_dir_cbin,
                                                                 annot_format='notmat')
        scribe_cbin = crowsetta.Transcriber(voc_format='notmat')
        self.annot_list_cbin = scribe_cbin.to_seq(file=self.annot_files_cbin)

        self.labelset_cbin = list('iabcdefghjk')

    def test_Vocalization_init(self):
        for spect_path, annot in zip(self.spect_list_mat, self.annot_list):
            spect_dict = loadmat(spect_path, squeeze_me=True)
            metaspect = MetaSpect.from_dict(spect_file_dict=spect_dict,
                                            freqbins_key='f',
                                            timebins_key='t',
                                            spect_key='s',
                                            timebin_dur=None,
                                            n_decimals_trunc=3)
            dur = metaspect.timebin_dur * metaspect.spect.shape[-1]
            voc = Vocalization(annot=annot,
                               duration=dur,
                               metaspect=metaspect,
                               spect_path=spect_path)
            for attr in ['annot', 'duration', 'spect_path', 'metaspect', 'audio', 'audio_path']:
                self.assertTrue(hasattr(voc, attr))
            self.assertTrue(voc.duration == dur)
            self.assertTrue(voc.spect_path == spect_path)
            self.assertTrue(voc.audio is None)
            self.assertTrue(voc.audio_path is None)

        for audio_path, annot in zip(self.audio_files_cbin, self.annot_list_cbin):
            fs, audio = load_cbin(audio_path)
            dur = audio.shape[-1] / fs
            voc = Vocalization(annot=annot,
                               duration=dur,
                               audio=audio,
                               audio_path=audio_path)
            for attr in ['annot', 'duration', 'spect_path', 'metaspect', 'audio', 'audio_path']:
                self.assertTrue(hasattr(voc, attr))
            self.assertTrue(voc.duration == dur)
            self.assertTrue(voc.spect_path is None)
            self.assertTrue(np.array_equal(voc.audio, audio))
            self.assertTrue(voc.audio_path == audio_path)

        with self.assertRaises(ValueError):
            # because we didn't specify audio or metaspect or audio_path or spect_path
            # notice we lazily re-use last value of annot and dur from loop above
            Vocalization(annot=annot,
                         duration=dur)

        with self.assertRaises(ValueError):
            # because we didn't specify spect path
            Vocalization(annot=annot,
                         duration=dur,
                         metaspect=metaspect)

        with self.assertRaises(ValueError):
            # because we didn't specify audio path
            Vocalization(annot=annot,
                         duration=dur,
                         audio=np.random.normal(size=(1000, 1)))

        # this should work, because we want to be able to have a Vocalization
        # without loading the spectrogram into it
        a_voc = Vocalization(annot=annot,
                             duration=dur,
                             spect_path=spect_path)
        for attr in ['annot', 'duration', 'spect_path', 'metaspect', 'audio', 'audio_path']:
            self.assertTrue(hasattr(a_voc, attr))
        self.assertTrue(a_voc.duration == dur)
        self.assertTrue(a_voc.metaspect is None)
        self.assertTrue(a_voc.spect_path == spect_path)
        self.assertTrue(a_voc.audio is None)
        self.assertTrue(a_voc.audio_path is None)

        # this should work, because we want to be able to have a Vocalization
        # without loading the audio into it
        a_voc = Vocalization(annot=annot,
                             duration=dur,
                             audio_path=self.audio_files_cbin[0])
        for attr in ['annot', 'duration', 'spect_path', 'metaspect', 'audio', 'audio_path']:
            self.assertTrue(hasattr(a_voc, attr))
        self.assertTrue(a_voc.duration == dur)
        self.assertTrue(a_voc.metaspect is None)
        self.assertTrue(a_voc.spect_path is None)
        self.assertTrue(a_voc.audio is None)
        self.assertTrue(a_voc.audio_path == self.audio_files_cbin[0])


if __name__ == '__main__':
    unittest.main()
