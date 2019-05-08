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
from vak.dataset.classes import VocalDataset, Vocalization, Spectrogram


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

    def _check_arr_files_returned_by_to_arr_files(self, arr_files):
        """assertions that are shared across unit tests for vak.dataset.audio.to_arr_files"""
        self.assertTrue(
            type(arr_files) == list
        )

        self.assertTrue(
            all([os.path.isfile(arr_file) for arr_file in arr_files])
        )

        # self.assertTrue(
        #     all([hasattr(voc, 'spect') for voc in vocal_dataset.voc_list])
        # )
        #
        # self.assertTrue(
        #     all([type(voc.spect) == Spectrogram for voc in vocal_dataset.voc_list])
        # )
        #
        # array_list_basenames = [os.path.basename(arr_path) for arr_path in self.array_list]
        # spect_files = [os.path.basename(voc.spect_file)
        #                for voc in vocal_dataset.voc_list]
        # self.assertTrue(
        #     all([spect_file in array_list_basenames for spect_file in spect_files])
        # )

        # if all assertTrues were True
        return True

    def test_audio_dir_annot_cbin(self):
        array_files = vak.dataset.audio.to_arr_files(audio_format='cbin',
                                                     spect_params=self.spect_params,
                                                     output_dir=self.tmp_output_dir,
                                                     audio_dir=self.audio_dir_cbin,
                                                     audio_files=None,
                                                     annot_list=self.annot_list_cbin,
                                                     audio_annot_map=None,
                                                     labelset=self.labelset_cbin,
                                                     skip_files_with_labels_not_in_labelset=True,
                                                     freqbins_key='f',
                                                     timebins_key='t',
                                                     spect_key='s')
        self.assertTrue(
            self._check_arr_files_returned_by_to_arr_files(array_files)
        )

    def test_audio_files_annot_cbin(self):
        array_files = vak.dataset.audio.to_arr_files(audio_format='cbin',
                                                     spect_params=self.spect_params,
                                                     output_dir=self.tmp_output_dir,
                                                     audio_dir=None,
                                                     audio_files=self.audio_files_cbin,
                                                     annot_list=self.annot_list_cbin,
                                                     audio_annot_map=None,
                                                     labelset=self.labelset_cbin,
                                                     skip_files_with_labels_not_in_labelset=True,
                                                     freqbins_key='f',
                                                     timebins_key='t',
                                                     spect_key='s')
        self.assertTrue(
            self._check_arr_files_returned_by_to_arr_files(array_files)
        )

    def test_audio_annot_map_cbin(self):
        audio_annot_map = dict(zip(self.audio_files_cbin, self.annot_list_cbin))
        array_files = vak.dataset.audio.to_arr_files(audio_format='cbin',
                                                     spect_params=self.spect_params,
                                                     output_dir=self.tmp_output_dir,
                                                     audio_dir=None,
                                                     audio_files=None,
                                                     annot_list=None,
                                                     audio_annot_map=audio_annot_map,
                                                     labelset=self.labelset_cbin,
                                                     skip_files_with_labels_not_in_labelset=True,
                                                     freqbins_key='f',
                                                     timebins_key='t',
                                                     spect_key='s')
        self.assertTrue(
            self._check_arr_files_returned_by_to_arr_files(array_files)
        )

    def test_bad_inputs_raise(self):
        # invalid audio format
        with self.assertRaises(ValueError):
            array_files = vak.dataset.audio.to_arr_files(audio_format='ape',
                                                         spect_params=self.spect_params,
                                                         output_dir=self.tmp_output_dir,
                                                         audio_dir=self.audio_dir_cbin,
                                                         audio_files=None,
                                                         annot_list=self.annot_list_cbin,
                                                         audio_annot_map=None,
                                                         labelset=self.labelset_cbin,
                                                         skip_files_with_labels_not_in_labelset=True,
                                                         freqbins_key='f',
                                                         timebins_key='t',
                                                         spect_key='s')

        # can't specify both dir and files
        with self.assertRaises(ValueError):
            array_files = vak.dataset.audio.to_arr_files(audio_format='cbin',
                                                         spect_params=self.spect_params,
                                                         output_dir=self.tmp_output_dir,
                                                         audio_dir=self.audio_dir_cbin,
                                                         audio_files=self.audio_files_cbin,
                                                         annot_list=self.annot_list_cbin,
                                                         audio_annot_map=None,
                                                         labelset=self.labelset_cbin,
                                                         skip_files_with_labels_not_in_labelset=True,
                                                         freqbins_key='f',
                                                         timebins_key='t',
                                                         spect_key='s')
        # can't specify both dir and audio_annot_map
        audio_annot_map = dict(zip(self.audio_files_cbin, self.annot_list_cbin))
        with self.assertRaises(ValueError):
            array_files = vak.dataset.audio.to_arr_files(audio_format='cbin',
                                                         spect_params=self.spect_params,
                                                         output_dir=self.tmp_output_dir,
                                                         audio_dir=self.audio_dir_cbin,
                                                         audio_files=None,
                                                         annot_list=None,
                                                         audio_annot_map=audio_annot_map,
                                                         labelset=self.labelset_cbin,
                                                         skip_files_with_labels_not_in_labelset=True,
                                                         freqbins_key='f',
                                                         timebins_key='t',
                                                         spect_key='s')

        # can't specify both list and audio_annot_map
        with self.assertRaises(ValueError):
            array_files = vak.dataset.audio.to_arr_files(audio_format='cbin',
                                                         spect_params=self.spect_params,
                                                         output_dir=self.tmp_output_dir,
                                                         audio_dir=None,
                                                         audio_files=self.audio_files_cbin,
                                                         annot_list=None,
                                                         audio_annot_map=audio_annot_map,
                                                         labelset=self.labelset_cbin,
                                                         skip_files_with_labels_not_in_labelset=True,
                                                         freqbins_key='f',
                                                         timebins_key='t',
                                                         spect_key='s')

        # can't specify both annotations list and audio_annot_map
        with self.assertRaises(ValueError):
            array_files = vak.dataset.audio.to_arr_files(audio_format='cbin',
                                                         spect_params=self.spect_params,
                                                         output_dir=self.tmp_output_dir,
                                                         audio_dir=None,
                                                         audio_files=None,
                                                         annot_list=self.annot_list_cbin,
                                                         audio_annot_map=audio_annot_map,
                                                         labelset=self.labelset_cbin,
                                                         skip_files_with_labels_not_in_labelset=True,
                                                         freqbins_key='f',
                                                         timebins_key='t',
                                                         spect_key='s')


if __name__ == '__main__':
    unittest.main()
