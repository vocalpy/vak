import os
import unittest
from glob import glob
import json

import numpy as np
from scipy.io import loadmat
import crowsetta

from vak.evfuncs import load_cbin
import vak.dataset.spect
import vak.dataset.annot
from vak.dataset.classes import VocalizationDataset, Vocalization, SpectrogramFile


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

        self.annot_files_cbin = vak.dataset.annot.files_from_dir(annot_dir=self.audio_dir_cbin,
                                                            annot_format='notmat')
        scribe_cbin = crowsetta.Transcriber(voc_format='notmat')
        self.annot_list_cbin = scribe_cbin.to_seq(file=self.annot_files_cbin)

        self.labelset_cbin = list('iabcdefghjk')

    def test_Spectrogram_init(self):
        for spect_path in self.spect_list_mat:
            spect_dict = loadmat(spect_path, squeeze_me=True)
            a_spect = SpectrogramFile(freq_bins=spect_dict['f'],
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

    def test_SpectrogramFile_from_dict(self):
        for spect_path in self.spect_list_mat:
            spect_dict = loadmat(spect_path, squeeze_me=True)
            spect_file = SpectrogramFile.from_dict(spect_file_dict=spect_dict,
                                                   freqbins_key='f',
                                                   timebins_key='t',
                                                   spect_key='s',
                                                   timebin_dur=None,
                                                   n_decimals_trunc=3)
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'spect']:
                self.assertTrue(hasattr(spect_file, attr))
                if attr in ['freq_bins', 'time_bins', 'spect']:
                    self.assertTrue(type(getattr(spect_file, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(spect_file, attr)) in (float, np.float16, np.float32, np.float64))

    def test_Vocalization_init(self):
        for spect_path, annot in zip(self.spect_list_mat, self.annot_list):
            spect_dict = loadmat(spect_path, squeeze_me=True)
            spect_file = SpectrogramFile.from_dict(spect_file_dict=spect_dict,
                                                   freqbins_key='f',
                                                   timebins_key='t',
                                                   spect_key='s',
                                                   timebin_dur=None,
                                                   n_decimals_trunc=3)
            dur = spect_file.timebin_dur * spect_file.spect.shape[-1]
            voc = Vocalization(annot=annot,
                               duration=dur,
                               spect_file=spect_file,
                               spect_path=spect_path)
            for attr in ['annot', 'duration', 'spect_path', 'spect_file', 'audio', 'audio_path']:
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
            for attr in ['annot', 'duration', 'spect_path', 'spect_file', 'audio', 'audio_path']:
                self.assertTrue(hasattr(voc, attr))
            self.assertTrue(voc.duration == dur)
            self.assertTrue(voc.spect_path is None)
            self.assertTrue(np.array_equal(voc.audio, audio))
            self.assertTrue(voc.audio_path == audio_path)

        with self.assertRaises(ValueError):
            # because we didn't specify audio or spect or audio_file or spect_file
            # notice we lazily re-use last value of annot and dur from loop above
            Vocalization(annot=annot,
                         duration=dur)

        with self.assertRaises(ValueError):
            # because we didn't specify spect path
            Vocalization(annot=annot,
                         duration=dur,
                         spect_file=spect_file)

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
        for attr in ['annot', 'duration', 'spect_path', 'spect_file', 'audio', 'audio_path']:
            self.assertTrue(hasattr(a_voc, attr))
        self.assertTrue(a_voc.duration == dur)
        self.assertTrue(a_voc.spect_file is None)
        self.assertTrue(a_voc.spect_path == spect_path)
        self.assertTrue(a_voc.audio is None)
        self.assertTrue(a_voc.audio_path is None)

        # this should work, because we want to be able to have a Vocalization
        # without loading the audio into it
        a_voc = Vocalization(annot=annot,
                             duration=dur,
                             audio_path=self.audio_files_cbin[0])
        for attr in ['annot', 'duration', 'spect_path', 'spect_file', 'audio', 'audio_path']:
            self.assertTrue(hasattr(a_voc, attr))
        self.assertTrue(a_voc.duration == dur)
        self.assertTrue(a_voc.spect_file is None)
        self.assertTrue(a_voc.spect_path is None)
        self.assertTrue(a_voc.audio is None)
        self.assertTrue(a_voc.audio_path == self.audio_files_cbin[0])

    def test_VocalizationDataset_init(self):
        voc_list = []
        for spect_path, annot in zip(self.spect_list_mat, self.annot_list):
            spect_dict = loadmat(spect_path, squeeze_me=True)
            spect_file = SpectrogramFile.from_dict(spect_file_dict=spect_dict,
                                                   freqbins_key='f',
                                                   timebins_key='t',
                                                   spect_key='s',
                                                   timebin_dur=None,
                                                   n_decimals_trunc=3)
            dur = spect_file.timebin_dur * spect_file.spect.shape[-1]
            voc = Vocalization(annot=annot,
                               duration=dur,
                               spect_file=spect_file,
                               spect_path=spect_path)
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

    def test_VocalizationDataset_json(self):
        vds = vak.dataset.spect.from_files(spect_format='mat',
                                           spect_dir=self.spect_dir_mat,
                                           annot_list=self.annot_list,
                                           load_spects=True)
        vds_json_str = vds.to_json(json_fname=None)
        vds_json = json.loads(vds_json_str)

        self.assertTrue(type(vds_json == dict))
        self.assertTrue('voc_list' in vds_json)
        voc_list = vds_json['voc_list']
        self.assertTrue(type(voc_list) == list)
        for voc in voc_list:
            for key in ['annot', 'duration', 'spect_path', 'spect_file', 'audio', 'audio_path']:
                self.assertTrue(key in voc)

        vds_from_json = VocalizationDataset.from_json(json_str=vds_json_str)
        self.assertTrue(type(vds_from_json) == VocalizationDataset)
        self.assertTrue(hasattr(vds, 'voc_list'))
        self.assertTrue(
            all([type(voc) == Vocalization for voc in vds.voc_list])
        )

    def test_VocalizationDataset_load_spects(self):
        vds = vak.dataset.spect.from_files(spect_format='mat',
                                           spect_dir=self.spect_dir_mat,
                                           annot_list=self.annot_list,
                                           load_spects=False)
        self.assertTrue(
            all([voc.spect_file is None for voc in vds.voc_list])
        )

        # check whether order of spect paths changes because we convert voc_list
        # to a dask bag to parallelize loading and order is usually preserved but this
        # is not guaranteed
        spect_paths_before = [voc.spect_path for voc in vds.voc_list]
        vds.load_spects()
        self.assertTrue(
            all([type(voc.spect_file) == SpectrogramFile for voc in vds.voc_list])
        )
        spect_paths_after = [voc.spect_path for voc in vds.voc_list]
        for before, after in zip(spect_paths_before, spect_paths_after):
            self.assertTrue(
                before == after
            )

    def test_VocalizationDataset_clear_spects(self):
        vds = vak.dataset.spect.from_files(spect_format='mat',
                                           spect_dir=self.spect_dir_mat,
                                           annot_list=self.annot_list,
                                           load_spects=True)
        self.assertTrue(
            all([type(voc.spect_file) == SpectrogramFile for voc in vds.voc_list])
        )

        vds.clear_spects()
        self.assertTrue(
            all([voc.spect_file is None for voc in vds.voc_list])
        )

    def test_VocalizationDataset_are_spects_loaded(self):
        vds = vak.dataset.spect.from_files(spect_format='mat',
                                           spect_dir=self.spect_dir_mat,
                                           annot_list=self.annot_list,
                                           load_spects=False)
        self.assertTrue(
            vds.are_spects_loaded() is False
        )

        vds.load_spects()
        self.assertTrue(
            vds.are_spects_loaded() is True
        )

    def test_VocalizationDataset_spects_list(self):
        vds = vak.dataset.spect.from_files(spect_format='mat',
                                           spect_dir=self.spect_dir_mat,
                                           annot_list=self.annot_list,
                                           load_spects=True)
        spects_list = vds.spects_list()
        self.assertTrue(
            type(spects_list == list)
        )
        self.assertTrue(
            all([type(spect) == np.ndarray for spect in spects_list])
        )


if __name__ == '__main__':
    unittest.main()
