import os
import unittest
from glob import glob
import json

import numpy as np
from scipy.io import loadmat
import crowsetta
import evfuncs

import vak.dataset.array
import vak.dataset.annot
from vak.dataset.classes import VocalizationDataset, Vocalization, Spectrogram


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
        self.array_dir_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        self.array_list_mat = glob(os.path.join(self.array_dir_mat, '*.mat'))

        self.annot_mat = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')
        self.scribe = crowsetta.Transcriber(voc_format='yarden')
        self.annot_list = self.scribe.to_seq(self.annot_mat)

        self.spect_params = dict(fft_size=512,
                                 step_size=64,
                                 freq_cutoffs=(500, 10000),
                                 thresh=6.25,
                                 transform_type='log_spect')

        # ---- cbins -------------------------------
        self.audio_dir_cbin = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        self.audio_files_cbin = glob(os.path.join(self.audio_dir_cbin, '*.cbin'))

        self.annot_files_cbin = vak.dataset.annot.files_from_dir(annot_dir=self.audio_dir_cbin,
                                                            annot_format='notmat')
        scribe_cbin = crowsetta.Transcriber(voc_format='notmat')
        self.annot_list_cbin = scribe_cbin.to_seq(file=self.annot_files_cbin)

        self.labelset_cbin = list('iabcdefghjk')

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
            voc = Vocalization(annotation=annot,
                               duration=dur,
                               spect=spect,
                               spect_file=arr_file)
            for attr in ['annotation', 'duration', 'spect', 'spect_file', 'audio', 'audio_file']:
                self.assertTrue(hasattr(voc, attr))
            self.assertTrue(voc.duration == dur)
            self.assertTrue(voc.spect_file == arr_file)
            self.assertTrue(voc.audio is None)
            self.assertTrue(voc.audio_file is None)

        for cbin_file, annot in zip(self.array_list_mat, self.annot_list):
            arr_file_dict = loadmat(arr_file, squeeze_me=True)
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

        with self.assertRaises(ValueError):
            # because we didn't specify audio or spect or audio_file or spect_file
            # notice we lazily re-use last value of annot and dur from loop above
            Vocalization(annotation=annot,
                         duration=dur)

        with self.assertRaises(ValueError):
            # because we didn't specify spect file
            Vocalization(annotation=annot,
                         duration=dur,
                         spect=spect)

        with self.assertRaises(ValueError):
            # because we didn't specify audio file
            Vocalization(annotation=annot,
                         duration=dur,
                         audio=np.random.normal(size=(1000, 1)))

        # this should work, because we want to be able to have a Vocalization
        # without loading the spectrogram into it
        a_voc = Vocalization(annotation=annot,
                             duration=dur,
                             spect_file=arr_file)
        for attr in ['annotation', 'duration', 'spect', 'spect_file', 'audio', 'audio_file']:
            self.assertTrue(hasattr(voc, attr))
        self.assertTrue(a_voc.duration == dur)
        self.assertTrue(a_voc.spect is None)
        self.assertTrue(a_voc.spect_file == arr_file)
        self.assertTrue(a_voc.audio is None)
        self.assertTrue(a_voc.audio_file is None)

        # this should work, because we want to be able to have a Vocalization
        # without loading the audio into it
        a_voc = Vocalization(annotation=annot,
                             duration=dur,
                             audio_file=self.audio_files_cbin[0])
        for attr in ['annotation', 'duration', 'spect', 'spect_file', 'audio', 'audio_file']:
            self.assertTrue(hasattr(voc, attr))
        self.assertTrue(a_voc.duration == dur)
        self.assertTrue(a_voc.spect is None)
        self.assertTrue(a_voc.spect_file is None)
        self.assertTrue(a_voc.audio is None)
        self.assertTrue(a_voc.audio_file == self.audio_files_cbin[0])

    def test_VocalDataset_init(self):
        voc_list = []
        for arr_file, annot in zip(self.array_list_mat, self.annot_list):
            arr_file_dict = loadmat(arr_file, squeeze_me=True)
            spect = Spectrogram.from_arr_file_dict(arr_file_dict=arr_file_dict,
                                                   freqbins_key='f',
                                                   timebins_key='t',
                                                   spect_key='s',
                                                   timebin_dur=None,
                                                   n_decimals_trunc=3)
            dur = spect.timebin_dur * spect.array.shape[-1]
            voc = Vocalization(annotation=annot,
                               duration=dur,
                               spect=spect,
                               spect_file=arr_file)
            voc_list.append(voc)

        vds = VocalizationDataset(voc_list=voc_list)
        self.assertTrue(type(vds) == VocalizationDataset)
        self.assertTrue(hasattr(vds, 'voc_list'))
        self.assertTrue(
            all([type(voc) == Vocalization for voc in vds.voc_list])
        )

    def _vocset_json_asserts(self, vocset_from_json):
        self.assertTrue(type(vocset_from_json == dict))
        self.assertTrue('voc_list' in vocset_from_json)
        voc_list = vocset_from_json['voc_list']
        self.assertTrue(type(voc_list) == list)

        # if all assertTrues are True
        return True

    def test_vocal_dataset_json(self):
        vds = vak.dataset.array.from_arr_files(array_format='mat',
                                               array_dir=self.array_dir_mat,
                                               annot_list=self.annot_list,
                                               load_spects=True)
        vds_json_str = vds.to_json(json_fname=None)
        vds_json = json.loads(vds_json_str)

        self.assertTrue(type(vds_json == dict))
        self.assertTrue('voc_list' in vds_json)
        voc_list = vds_json['voc_list']
        self.assertTrue(type(voc_list) == list)
        for voc in voc_list:
            for key in ['annotation', 'duration', 'spect', 'spect_file', 'audio', 'audio_file']:
                self.assertTrue(key in voc)

        vds_from_json = VocalizationDataset.from_json(json_str=vds_json_str)
        self.assertTrue(type(vds_from_json) == VocalizationDataset)
        self.assertTrue(hasattr(vds, 'voc_list'))
        self.assertTrue(
            all([type(voc) == Vocalization for voc in vds.voc_list])
        )


if __name__ == '__main__':
    unittest.main()
