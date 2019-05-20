import os
import unittest
from glob import glob
import tempfile
import shutil
from collections import namedtuple
import copy

import numpy as np
import crowsetta

from vak.config.spectrogram import SpectConfig
from vak.dataset.annot import files_from_dir
import vak.dataset.audio
from vak.dataset.classes import VocalizationDataset, Vocalization, MetaSpect


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE,
                             '..',
                             '..',
                             'test_data')
SETUP_SCRIPTS_DIR = os.path.join(HERE,
                                 '..',
                                 '..',
                                 'setup_scripts')


class TestAudio(unittest.TestCase):
    def setUp(self):
        self.spect_params = dict(fft_size=512,
                                 step_size=64,
                                 freq_cutoffs=(500, 10000),
                                 thresh=6.25,
                                 transform_type='log_spect')

        self.tmp_output_dir = tempfile.mkdtemp()

        # ---- cbins -------------------------------
        self.audio_dir_cbin = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        self.audio_files_cbin = glob(os.path.join(self.audio_dir_cbin, '*.cbin'))

        annot_files_cbin = files_from_dir(annot_dir=self.audio_dir_cbin, annot_format='notmat')
        scribe_cbin = crowsetta.Transcriber(voc_format='notmat')
        self.annot_list_cbin = scribe_cbin.to_seq(file=annot_files_cbin)

        self.labelset_cbin = list('iabcdefghjk')

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)

    def _check_spect_files_returned_by_to_spect_files(self, spect_files):
        """assertions that are shared across unit tests for vak.dataset.audio.to_spect"""
        self.assertTrue(
            type(spect_files) == list
        )

        self.assertTrue(
            all([os.path.isfile(spect_file) for spect_file in spect_files])
        )

        for spect_file in spect_files:
            spect_dict = np.load(spect_file)
            for key in ['s', 'f', 't']:
                self.assertTrue(key in spect_dict)
                self.assertTrue(type(spect_dict[key]) == np.ndarray)

        # if all assertTrues were True
        return True

    def test_audio_dir_annot_cbin(self):
        spect_files = vak.dataset.audio.to_spect(audio_format='cbin',
                                                 spect_params=self.spect_params,
                                                 output_dir=self.tmp_output_dir,
                                                 audio_dir=self.audio_dir_cbin,
                                                 audio_files=None,
                                                 annot_list=self.annot_list_cbin,
                                                 audio_annot_map=None,
                                                 labelset=self.labelset_cbin,
                                                 freqbins_key='f',
                                                 timebins_key='t',
                                                 spect_key='s')
        self.assertTrue(
            self._check_spect_files_returned_by_to_spect_files(spect_files)
        )

    def test_audio_files_annot_cbin(self):
        spect_files = vak.dataset.audio.to_spect(audio_format='cbin',
                                                 spect_params=self.spect_params,
                                                 output_dir=self.tmp_output_dir,
                                                 audio_dir=None,
                                                 audio_files=self.audio_files_cbin,
                                                 annot_list=self.annot_list_cbin,
                                                 audio_annot_map=None,
                                                 labelset=self.labelset_cbin,
                                                 freqbins_key='f',
                                                 timebins_key='t',
                                                 spect_key='s')
        self.assertTrue(
            self._check_spect_files_returned_by_to_spect_files(spect_files)
        )

    def test_audio_annot_map_cbin(self):
        audio_annot_map = dict(zip(self.audio_files_cbin, self.annot_list_cbin))
        spect_files = vak.dataset.audio.to_spect(audio_format='cbin',
                                                 spect_params=self.spect_params,
                                                 output_dir=self.tmp_output_dir,
                                                 audio_dir=None,
                                                 audio_files=None,
                                                 annot_list=None,
                                                 audio_annot_map=audio_annot_map,
                                                 labelset=self.labelset_cbin,
                                                 freqbins_key='f',
                                                 timebins_key='t',
                                                 spect_key='s')
        self.assertTrue(
            self._check_spect_files_returned_by_to_spect_files(spect_files)
        )

    def test_audio_dir_without_annot(self):
        # make sure we can make a spectrograms from audio files without annotations,
        # e.g. if we're going to predict the annotations using the spectrograms
        spect_files = vak.dataset.audio.to_spect(audio_format='cbin',
                                                 spect_params=self.spect_params,
                                                 output_dir=self.tmp_output_dir,
                                                 audio_dir=self.audio_dir_cbin,
                                                 audio_files=None,
                                                 annot_list=None,
                                                 audio_annot_map=None,
                                                 labelset=None,
                                                 freqbins_key='f',
                                                 timebins_key='t',
                                                 spect_key='s')
        self.assertTrue(
            self._check_spect_files_returned_by_to_spect_files(spect_files)
        )

    def test_bad_inputs_raise(self):
        # must specify one of: audio files, audio list, or audio files/annotations mapping
        with self.assertRaises(ValueError):
            spect_files = vak.dataset.audio.to_spect(audio_format='ape',
                                                     spect_params=self.spect_params,
                                                     output_dir=self.tmp_output_dir,
                                                     audio_dir=None,
                                                     audio_files=None,
                                                     annot_list=self.annot_list_cbin,
                                                     audio_annot_map=None,
                                                     labelset=self.labelset_cbin,
                                                     freqbins_key='f',
                                                     timebins_key='t',
                                                     spect_key='s')

        # invalid audio format
        with self.assertRaises(ValueError):
            spect_files = vak.dataset.audio.to_spect(audio_format='ape',
                                                     spect_params=self.spect_params,
                                                     output_dir=self.tmp_output_dir,
                                                     audio_dir=self.audio_dir_cbin,
                                                     audio_files=None,
                                                     annot_list=self.annot_list_cbin,
                                                     audio_annot_map=None,
                                                     labelset=self.labelset_cbin,
                                                     freqbins_key='f',
                                                     timebins_key='t',
                                                     spect_key='s')

        # can't specify both dir and files
        with self.assertRaises(ValueError):
            spect_files = vak.dataset.audio.to_spect(audio_format='cbin',
                                                     spect_params=self.spect_params,
                                                     output_dir=self.tmp_output_dir,
                                                     audio_dir=self.audio_dir_cbin,
                                                     audio_files=self.audio_files_cbin,
                                                     annot_list=self.annot_list_cbin,
                                                     audio_annot_map=None,
                                                     labelset=self.labelset_cbin,
                                                     freqbins_key='f',
                                                     timebins_key='t',
                                                     spect_key='s')
        # can't specify both dir and audio_annot_map
        audio_annot_map = dict(zip(self.audio_files_cbin, self.annot_list_cbin))
        with self.assertRaises(ValueError):
            spect_files = vak.dataset.audio.to_spect(audio_format='cbin',
                                                     spect_params=self.spect_params,
                                                     output_dir=self.tmp_output_dir,
                                                     audio_dir=self.audio_dir_cbin,
                                                     audio_files=None,
                                                     annot_list=None,
                                                     audio_annot_map=audio_annot_map,
                                                     labelset=self.labelset_cbin,
                                                     freqbins_key='f',
                                                     timebins_key='t',
                                                     spect_key='s')

        # can't specify both list and audio_annot_map
        with self.assertRaises(ValueError):
            spect_files = vak.dataset.audio.to_spect(audio_format='cbin',
                                                     spect_params=self.spect_params,
                                                     output_dir=self.tmp_output_dir,
                                                     audio_dir=None,
                                                     audio_files=self.audio_files_cbin,
                                                     annot_list=None,
                                                     audio_annot_map=audio_annot_map,
                                                     labelset=self.labelset_cbin,
                                                     freqbins_key='f',
                                                     timebins_key='t',
                                                     spect_key='s')

        # can't specify both annotations list and audio_annot_map
        with self.assertRaises(ValueError):
            spect_files = vak.dataset.audio.to_spect(audio_format='cbin',
                                                     spect_params=self.spect_params,
                                                     output_dir=self.tmp_output_dir,
                                                     audio_dir=None,
                                                     audio_files=None,
                                                     annot_list=self.annot_list_cbin,
                                                     audio_annot_map=audio_annot_map,
                                                     labelset=self.labelset_cbin,
                                                     freqbins_key='f',
                                                     timebins_key='t',
                                                     spect_key='s')


if __name__ == '__main__':
    unittest.main()
