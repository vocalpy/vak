import os
from pathlib import Path
import unittest
from glob import glob
import tempfile
import shutil

import numpy as np
import crowsetta

from vak.io.annotation import files_from_dir
import vak.io.audio


HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestAudio(unittest.TestCase):
    def setUp(self):
        self.spect_params = dict(fft_size=512,
                                 step_size=64,
                                 freq_cutoffs=(500, 10000),
                                 thresh=6.25,
                                 transform_type='log_spect')

        self.tmp_output_dir = tempfile.mkdtemp()

        # ---- cbins -------------------------------
        self.audio_dir_cbin = TEST_DATA_DIR.joinpath('cbins', 'gy6or6', '032312')
        self.audio_files_cbin = sorted(
            list(self.audio_dir_cbin.glob('*.cbin'))
        )
        self.audio_files_cbin = [str(path) for path in self.audio_files_cbin]

        self.annot_files_cbin = files_from_dir(annot_dir=self.audio_dir_cbin,
                                               annot_format='notmat')
        scribe_cbin = crowsetta.Transcriber(annot_format='notmat')
        self.annot_list_cbin = scribe_cbin.from_file(annot_file=self.annot_files_cbin)

        self.labelset_cbin = set(list('iabcdefghjk'))

        # sort annotation, audio into lists so we can verify labelset works
        # "good" = all labels in annotation are in labelset
        self.good = [(annot_file, Path(annot.audio_file).name)
                     for annot_file, annot in zip(self.annot_files_cbin,
                                                  self.annot_list_cbin)
                     if set(annot.seq.labels).issubset(self.labelset_cbin)]

        # "bad" = has labels not in labelset
        self.bad = [(annot_file, Path(annot.audio_file).name)
                    for annot_file, annot in zip(self.annot_files_cbin,
                                             self.annot_list_cbin)
                    if not set(annot.seq.labels).issubset(self.labelset_cbin)]

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)

    def _check_spect_files(self, spect_files, labelset):
        """assertions that are shared across unit tests
        for vak.io.audio.to_spect"""
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

        source_audio_files = [spect_file.replace('.spect.npz', '')
                              for spect_file in spect_files]
        if labelset:
            # since we supplied labelset, only audio files whose annotation
            # has only labels in labelset should have a corresponding spect file
            good_audio_files = [good_tup[1] for good_tup in self.good]
            for good_audio_file in good_audio_files:
                self.assertTrue(
                    any([good_audio_file in s for s in source_audio_files])
                )
            bad_audio_files = [bad_tup[1] for bad_tup in self.bad]
            for bad_audio_file in bad_audio_files:
                self.assertTrue(
                    not any([bad_audio_file in s for s in source_audio_files])
                )
        else:
            # since we didn't supply labelset, all audio files should have
            # a corresponding spect file
            all_audio_files = [tup[1] for tup in self.good + self.bad]
            for audio_file in all_audio_files:
                self.assertTrue(
                    any([audio_file in s for s in source_audio_files])
                )

        # if all assertTrues were True
        return True

    def test_audio_dir_annot_cbin_with_labelset(self):
        spect_files = vak.io.audio.to_spect(audio_format='cbin',
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
            self._check_spect_files(spect_files,
                                    labelset=self.labelset_cbin)
        )

    def test_audio_dir_annot_cbin_no_labelset(self):
        spect_files = vak.io.audio.to_spect(audio_format='cbin',
                                            spect_params=self.spect_params,
                                            output_dir=self.tmp_output_dir,
                                            audio_dir=self.audio_dir_cbin,
                                            audio_files=None,
                                            annot_list=self.annot_list_cbin,
                                            audio_annot_map=None,
                                            labelset=None,
                                            freqbins_key='f',
                                            timebins_key='t',
                                            spect_key='s')
        self.assertTrue(
            self._check_spect_files(spect_files,
                                    labelset=None)
        )

    def test_audio_files_annot_cbin(self):
        spect_files = vak.io.audio.to_spect(audio_format='cbin',
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
            self._check_spect_files(spect_files,
                                    labelset=self.labelset_cbin)
        )

    def test_audio_files_annot_cbin_no_labelset(self):
        spect_files = vak.io.audio.to_spect(audio_format='cbin',
                                            spect_params=self.spect_params,
                                            output_dir=self.tmp_output_dir,
                                            audio_dir=None,
                                            audio_files=self.audio_files_cbin,
                                            annot_list=self.annot_list_cbin,
                                            audio_annot_map=None,
                                            labelset=None,
                                            freqbins_key='f',
                                            timebins_key='t',
                                            spect_key='s')
        self.assertTrue(
            self._check_spect_files(spect_files,
                                    labelset=None)
        )

    def test_audio_annot_map_cbin(self):
        audio_annot_map = dict(zip(self.audio_files_cbin, self.annot_list_cbin))
        spect_files = vak.io.audio.to_spect(audio_format='cbin',
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
            self._check_spect_files(spect_files,
                                    labelset=self.labelset_cbin)
        )

    def test_audio_annot_map_cbin_no_labelset(self):
        audio_annot_map = dict(zip(self.audio_files_cbin, self.annot_list_cbin))
        spect_files = vak.io.audio.to_spect(audio_format='cbin',
                                            spect_params=self.spect_params,
                                            output_dir=self.tmp_output_dir,
                                            audio_dir=None,
                                            audio_files=None,
                                            annot_list=None,
                                            audio_annot_map=audio_annot_map,
                                            labelset=None,
                                            freqbins_key='f',
                                            timebins_key='t',
                                            spect_key='s')
        self.assertTrue(
            self._check_spect_files(spect_files,
                                    labelset=None)
        )

    def test_audio_dir_without_annot(self):
        # make sure we can make a spectrograms from audio files without annotations,
        # e.g. if we're going to predict the annotations using the spectrograms
        spect_files = vak.io.audio.to_spect(audio_format='cbin',
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
            self._check_spect_files(spect_files, labelset=None)
        )

    def test_bad_inputs_raise(self):
        # must specify one of: audio files, audio list, or audio files/annotations mapping
        with self.assertRaises(ValueError):
            vak.io.audio.to_spect(audio_format='ape',
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
            vak.io.audio.to_spect(audio_format='ape',
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
            vak.io.audio.to_spect(audio_format='cbin',
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
            vak.io.audio.to_spect(audio_format='cbin',
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
            vak.io.audio.to_spect(audio_format='cbin',
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
            vak.io.audio.to_spect(audio_format='cbin',
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
