import os
import unittest
from glob import glob
import json

import numpy as np
from scipy.io import loadmat
import crowsetta

import vak.dataset.array
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


class TestClasses(unittest.TestCase):
    def setUp(self):
        self.array_dir_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb11', 'spect')
        self.array_list_mat = glob(os.path.join(self.array_dir_mat, '*.mat'))

        self.annot_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb11', 'llb11_annot_subset.mat')
        self.scribe = crowsetta.Transcriber(voc_format='yarden')
        self.annot_list = self.scribe.to_seq(self.annot_mat)

    def test_Spectrogram_init(self):
        for arr_file in self.array_list_mat:
            arr = loadmat(arr_file, squeeze_me=True)
            a_spect = Spectrogram(freq_bins=arr['f'],
                                  time_bins=arr['t'],
                                  timebin_dur=0.002,
                                  array=arr['s']
                                  )
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'array']:
                self.assertTrue(hasattr(a_spect, attr))
                if attr in ['freq_bins', 'time_bins', 'array']:
                    self.assertTrue(type(getattr(a_spect, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(a_spect, attr)) == float)

    def test_Spectrogram_from_arr_file_dict(self):
        for arr_file in self.array_list_mat:
            arr_file_dict = loadmat(arr_file, squeeze_me=True)
            a_spect = Spectrogram.from_arr_file_dict(arr_file_dict=arr_file_dict,
                                                     freqbins_key='f',
                                                     timebins_key='t',
                                                     spect_key='s',
                                                     timebin_dur=None,
                                                     n_decimals_trunc=3)
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'array']:
                self.assertTrue(hasattr(a_spect, attr))
                if attr in ['freq_bins', 'time_bins', 'array']:
                    self.assertTrue(type(getattr(a_spect, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(a_spect, attr)) in (float, np.float16, np.float32, np.float64))

    def test_Vocalization_init(self):
        for arr_file, annot in zip(self.array_list_mat, self.annot_list):
            arr_file_dict = loadmat(arr_file, squeeze_me=True)
            spect = Spectrogram.from_arr_file_dict(arr_file_dict=arr_file_dict,
                                                   freqbins_key='f',
                                                   timebins_key='t',
                                                   spect_key='s',
                                                   timebin_dur=None,
                                                   n_decimals_trunc=3)
            dur = spect.timebin_dur * spect.array.shape[-1]
            voc = Vocalization(annotation=self.annot_list[0],
                               duration=dur,
                               spect=spect,
                               spect_file=arr_file)
            for attr in ['annotation', 'duration', 'spect', 'spect_file', 'audio', 'audio_file']:
                self.assertTrue(hasattr(voc, attr))
            self.assertTrue(voc.duration == dur)
            self.assertTrue(voc.spect_file == arr_file)
            self.assertTrue(voc.audio is None)
            self.assertTrue(voc.audio_file is None)

    # def test_VocalDataset(self):
    #     raise NotImplementedError

    def _vocset_json_asserts(self, vocset_from_json):
        self.assertTrue(type(vocset_from_json == dict))
        self.assertTrue('voc_list' in vocset_from_json)
        voc_list = vocset_from_json['voc_list']
        self.assertTrue(type(voc_list) == list)

        # if all assertTrues are True
        return True

    def test_vocal_dataset_to_json(self):
        vocset = vak.dataset.array.from_arr_files(array_format='mat',
                                                  array_dir=self.array_dir_mat,
                                                  annot_list=self.annot_list,
                                                  load_spects=True)
        vocset_json = vocset.to_json(json_fname=None)
        vocset_from_json = json.loads(vocset_json)
        self.assertTrue(self._vocset_json_asserts(vocset_from_json))


if __name__ == '__main__':
    unittest.main()
