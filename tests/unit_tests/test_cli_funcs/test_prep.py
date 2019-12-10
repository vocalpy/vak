"""tests for vak.cli.prep module"""
import os
import tempfile
import shutil
from glob import glob
import unittest
from math import isclose

import crowsetta

import vak.cli.prep
from vak.config.spectrogram import SpectConfig
from vak.io import Dataset
import vak.io

HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')


class TestPrep(unittest.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()
        a_config = os.path.join(TEST_DATA_DIR, 'configs', 'test_learncurve_config.ini')
        self.tmp_config_path = os.path.join(TEST_DATA_DIR, 'configs', 'tmp_config.ini')
        shutil.copy(a_config, self.tmp_config_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)
        os.remove(self.tmp_config_path)

    def _check_output(self,
                      data_dir,
                      labelset,
                      audio_format,
                      spect_format,
                      annot_format,
                      annot_file,
                      vds_paths,
                      num_expected_paths,
                      splits=None,
                      specd_durs=None):
        self.assertTrue(len(vds_paths) == num_expected_paths)

        # check that all files from data_dir that should've gone into dataset
        # actually made it into dataset
        if audio_format:
            data_files_from_dir = vak.io.audio.files_from_dir(data_dir, audio_format)
        elif spect_format:
            data_files_from_dir = vak.utils.general._files_from_dir(data_dir, spect_format)

        if num_expected_paths == 1:
            vds = Dataset.from_json(json_fname=vds_paths[0])
            if audio_format:
                data_files_in_vds = [voc.audio_path for voc in vds.voc_list]
            elif spect_format:
                data_files_in_vds = [voc.spect_path for voc in vds.voc_list]

            if labelset is None:
                self.assertTrue(data_files_from_dir == data_files_in_vds)
            else:
                scribe = crowsetta.Transcriber(voc_format=annot_format)
                if annot_file:
                    annot_list = scribe.to_seq(file=annot_file)
                else:
                    annot_files = vak.io.annotation.files_from_dir(annot_dir=data_dir, annot_format=annot_format)
                    annot_list = scribe.to_seq(file=annot_files)
                for data_file, annot in zip(data_files_from_dir, annot_list):
                    if set(annot.labels).issubset(labelset):
                        self.assertTrue(
                            data_file in data_files_in_vds
                        )
                    else:
                        self.assertTrue(
                            data_file not in data_files_in_vds
                        )

        # if we split the dataset, make sure the split worked
        if splits and specd_durs:
            for split, specd_dur in zip(splits, specd_durs):
                path = [path for path in vds_paths if split in path]
                self.assertTrue(len(path) == 1)
                path = path[0]
                if specd_dur > 0:
                    vds_loaded = Dataset.from_json(json_fname=path)
                    total_dur = sum([voc.duration for voc in vds_loaded.voc_list])
                    self.assertTrue((total_dur >= specd_dur))

                elif specd_dur == -1:
                    vds_loaded = Dataset.from_json(json_fname=path)
                    total_dur = sum([voc.duration for voc in vds_loaded.voc_list])
                    source_vds_path = [path for path in vds_paths if 'test' not in path and 'train' not in path][0]
                    source_vds = Dataset.from_json(json_fname=source_vds_path)
                    source_dur = sum([voc.duration for voc in source_vds.voc_list])

                    if split == 'train':
                        test_path = [path for path in vds_paths if 'test' in path][0]
                        test_vds = Dataset.from_json(json_fname=test_path)
                        test_dur = sum([voc.duration for voc in test_vds.voc_list])
                        self.assertTrue(
                            isclose(total_dur, source_dur - test_dur)
                        )
                    elif split == 'test':
                        train_path = [path for path in vds_paths if 'train' in path][0]
                        train_vds = Dataset.from_json(json_fname=train_path)
                        train_dur = sum([voc.duration for voc in train_vds.voc_list])
                        self.assertTrue(
                            isclose(total_dur, source_dur - train_dur)
                        )

        return True

    def test_prep_with_audio_cbin_no_split(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                   transform_type='log_spect')
        labelset = set(list('iabcdefghjk'))

        annot_format = 'notmat'
        audio_format = 'cbin'
        spect_format = None
        annot_file = None

        train_dur = None
        val_dur = None
        test_dur = None

        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     train_dur=train_dur,
                     val_dur=val_dur,
                     test_dur=test_dur,
                     config_file=self.tmp_config_path,
                     annot_format=annot_format,
                     output_dir=self.tmp_output_dir,
                     audio_format=audio_format,
                     spect_format=spect_format,
                     annot_file=annot_file,
                     spect_params=spect_params)

        vds_paths = glob(os.path.join(self.tmp_output_dir, '*vds.json'))

        self.assertTrue(
            self._check_output(data_dir,
                               labelset,
                               audio_format,
                               spect_format,
                               annot_format,
                               annot_file,
                               vds_paths,
                               num_expected_paths=1)
        )

    def test_prep_with_audio_cbin_split_with_train(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                   transform_type='log_spect')
        labelset = list('iabcdefghjk')

        annot_format = 'notmat'
        audio_format = 'cbin'
        spect_format = None
        annot_file = None

        train_dur = 35
        val_dur = None
        test_dur = None

        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     train_dur=train_dur,
                     val_dur=val_dur,
                     test_dur=test_dur,
                     config_file=self.tmp_config_path,
                     annot_format=annot_format,
                     output_dir=self.tmp_output_dir,
                     audio_format=audio_format,
                     spect_format=spect_format,
                     annot_file=annot_file,
                     spect_params=spect_params)

        vds_paths = glob(os.path.join(self.tmp_output_dir, '*vds.json'))
        self.assertTrue(
            self._check_output(data_dir,
                               labelset,
                               audio_format,
                               spect_format,
                               annot_format,
                               annot_file,
                               vds_paths,
                               num_expected_paths=3,
                               splits=['train', 'test'],
                               specd_durs=[train_dur, -1]
                               )
        )

    def test_prep_with_audio_cbin_split_with_test(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                   transform_type='log_spect')
        labelset = list('iabcdefghjk')

        annot_format = 'notmat'
        audio_format = 'cbin'
        spect_format = None
        annot_file = None

        train_dur = None
        val_dur = None
        test_dur = 35

        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     train_dur=train_dur,
                     val_dur=val_dur,
                     test_dur=test_dur,
                     config_file=self.tmp_config_path,
                     annot_format=annot_format,
                     output_dir=self.tmp_output_dir,
                     audio_format=audio_format,
                     spect_format=spect_format,
                     annot_file=annot_file,
                     spect_params=spect_params)

        vds_paths = glob(os.path.join(self.tmp_output_dir, '*vds.json'))
        self.assertTrue(
            self._check_output(data_dir,
                               labelset,
                               audio_format,
                               spect_format,
                               annot_format,
                               annot_file,
                               vds_paths,
                               num_expected_paths=3,
                               splits=['train', 'test'],
                               specd_durs=[-1, test_dur])
        )

    def test_prep_with_audio_cbin_train_test_val_split(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                   transform_type='log_spect')
        labelset = list('iabcdefghjk')

        annot_format = 'notmat'
        audio_format = 'cbin'
        spect_format = None
        annot_file = None

        train_dur = 35
        val_dur = 20
        test_dur = 35

        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     train_dur=train_dur,
                     val_dur=val_dur,
                     test_dur=test_dur,
                     config_file=self.tmp_config_path,
                     annot_format=annot_format,
                     output_dir=self.tmp_output_dir,
                     audio_format=audio_format,
                     spect_format=spect_format,
                     annot_file=annot_file,
                     spect_params=spect_params)

        vds_paths = glob(os.path.join(self.tmp_output_dir, '*vds.json'))
        self.assertTrue(
            self._check_output(data_dir,
                               labelset,
                               audio_format,
                               spect_format,
                               annot_format,
                               annot_file,
                               vds_paths,
                               num_expected_paths=4,
                               splits=['train', 'val', 'test'],
                               specd_durs=[train_dur, val_dur, test_dur])
        )

    def test_prep_with_spect_mat_no_split(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        labelset = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

        annot_format = 'yarden'
        audio_format = None
        spect_format = 'mat'
        annot_file = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')

        train_dur = None
        val_dur = None
        test_dur = None

        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     train_dur=train_dur,
                     val_dur=val_dur,
                     test_dur=test_dur,
                     config_file=self.tmp_config_path,
                     annot_format=annot_format,
                     output_dir=self.tmp_output_dir,
                     audio_format=audio_format,
                     spect_format=spect_format,
                     annot_file=annot_file,
                     spect_params=None)

        vds_paths = glob(os.path.join(self.tmp_output_dir, '*vds.json'))
        self.assertTrue(
            self._check_output(data_dir,
                               labelset,
                               audio_format,
                               spect_format,
                               annot_format,
                               annot_file,
                               vds_paths,
                               num_expected_paths=1)
        )

    def test_prep_with_spect_mat_train(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        labelset = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

        annot_format = 'yarden'
        audio_format = None
        spect_format = 'mat'
        annot_file = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')

        train_dur = 200
        val_dur = None
        test_dur = None

        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     train_dur=train_dur,
                     val_dur=val_dur,
                     test_dur=test_dur,
                     config_file=self.tmp_config_path,
                     annot_format=annot_format,
                     output_dir=self.tmp_output_dir,
                     audio_format=audio_format,
                     spect_format=spect_format,
                     annot_file=annot_file,
                     spect_params=None)

        vds_paths = glob(os.path.join(self.tmp_output_dir, '*vds.json'))
        self.assertTrue(
            self._check_output(data_dir,
                               labelset,
                               audio_format,
                               spect_format,
                               annot_format,
                               annot_file,
                               vds_paths,
                               num_expected_paths=3,
                               splits=['train', 'test'],
                               specd_durs=[train_dur, -1])
        )

    def test_prep_with_spect_mat_test(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        labelset = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

        annot_format = 'yarden'
        audio_format = None
        spect_format = 'mat'
        annot_file = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')

        train_dur = None
        val_dur = None
        test_dur = 200

        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     train_dur=train_dur,
                     val_dur=val_dur,
                     test_dur=test_dur,
                     config_file=self.tmp_config_path,
                     annot_format=annot_format,
                     output_dir=self.tmp_output_dir,
                     audio_format=audio_format,
                     spect_format=spect_format,
                     annot_file=annot_file,
                     spect_params=None)

        vds_paths = glob(os.path.join(self.tmp_output_dir, '*vds.json'))
        self.assertTrue(
            self._check_output(data_dir,
                               labelset,
                               audio_format,
                               spect_format,
                               annot_format,
                               annot_file,
                               vds_paths,
                               num_expected_paths=3,
                               splits=['train', 'test'],
                               specd_durs=[-1, test_dur])
        )

    def test_prep_with_spect_mat_train_val_test_split(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'spect')
        labelset = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

        annot_format = 'yarden'
        audio_format = None
        spect_format = 'mat'
        annot_file = os.path.join(TEST_DATA_DIR, 'mat', 'llb3', 'llb3_annot_subset.mat')

        train_dur = 200
        val_dur = 100
        test_dur = 200

        vak.cli.prep(labelset=labelset,
                     data_dir=data_dir,
                     train_dur=train_dur,
                     val_dur=val_dur,
                     test_dur=test_dur,
                     config_file=self.tmp_config_path,
                     annot_format=annot_format,
                     output_dir=self.tmp_output_dir,
                     audio_format=audio_format,
                     spect_format=spect_format,
                     annot_file=annot_file,
                     spect_params=None)

        vds_paths = glob(os.path.join(self.tmp_output_dir, '*vds.json'))
        self.assertTrue(
            self._check_output(data_dir,
                               labelset,
                               audio_format,
                               spect_format,
                               annot_format,
                               annot_file,
                               vds_paths,
                               num_expected_paths=4,
                               splits=['train', 'val', 'test'],
                               specd_durs=[train_dur, val_dur, test_dur])
        )

    def test_prep_with_just_val_dur_raises(self):
        data_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        spect_params = SpectConfig(fft_size=512, step_size=64, freq_cutoffs=(500, 10000), thresh=6.25,
                                   transform_type='log_spect')
        labelset = list('iabcdefghjk')

        train_dur = None
        val_dur = 20
        test_dur = None

        with self.assertRaises(ValueError):
            vak.cli.prep(labelset=labelset,
                         data_dir=data_dir,
                         train_dur=train_dur,
                         val_dur=val_dur,
                         test_dur=test_dur,
                         config_file=self.tmp_config_path,
                         annot_format='notmat',
                         output_dir=self.tmp_output_dir,
                         audio_format='cbin',
                         spect_format=None,
                         annot_file=None,
                         spect_params=spect_params)


if __name__ == '__main__':
    unittest.main()
