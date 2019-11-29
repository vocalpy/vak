from glob import glob
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from scipy.io import loadmat
import crowsetta

import vak.dataset.dataframe
import vak.dataset.annotation
from vak.dataset.classes import Dataset, Vocalization, MetaSpect

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.spect_dir_mat = TEST_DATA_DIR.joinpath('mat', 'llb3', 'spect')
        self.spect_list_mat = list(self.spect_dir_mat.glob('*.mat'))
        self.spect_list_mat = [str(path) for path in self.spect_list_mat]

        self.annot_mat = str(TEST_DATA_DIR.joinpath('mat', 'llb3',
                                                    'llb3_annot_subset.mat'))
        scribe_mat = crowsetta.Transcriber(voc_format='yarden')
        self.annot_list_mat = scribe_mat.to_seq(self.annot_mat)
        self.labelset_mat = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}

        self.spect_params = dict(fft_size=512,
                                 step_size=64,
                                 freq_cutoffs=(500, 10000),
                                 thresh=6.25,
                                 transform_type='log_spect')

        # ---- in .npz files, made from .cbin audio files -------------------------------
        self.labelset_cbin = list('iabcdefghjk')

        self.spect_dir_npz = list(TEST_DATA_DIR.joinpath('vds').glob(
            'spectrograms_generated*')
        )
        self.spect_dir_npz = self.spect_dir_npz[0]
        self.spect_list_npz = list(self.spect_dir_npz.glob('*.spect.npz'))
        self.spect_list_npz = [str(path) for path in self.spect_list_npz]

        # now that we have .npz file list, use that to filter .not.mat list which we need when calling from_files
        self.audio_dir_cbin = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        self.audio_files_cbin = glob(os.path.join(self.audio_dir_cbin, '*.cbin'))
        self.annot_files_cbin = vak.dataset.annotation.files_from_dir(annot_dir=self.audio_dir_cbin,
                                                                      annot_format='notmat')
        self.annot_files_cbin = [annot_file_cbin
                                 for annot_file_cbin in self.annot_files_cbin
                                 if any([Path(annot_file_cbin).name.replace('.not.mat', '') in npz
                                         for npz in self.spect_list_npz])
                                 ]
        scribe_cbin = crowsetta.Transcriber(voc_format='notmat')
        self.annot_list_cbin = scribe_cbin.to_seq(file=self.annot_files_cbin)

    def test_Dataset_init(self):
        voc_list = []
        for spect_path, annot in zip(self.spect_list_mat, self.annot_list_mat):
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
            voc_list.append(voc)

        vds = Dataset(voc_list=voc_list)
        self.assertTrue(type(vds) == Dataset)
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

    def test_Dataset_json(self):
        vds = vak.dataset.dataframe.from_files(spect_format='mat',
                                               spect_dir=self.spect_dir_mat,
                                               annot_list=self.annot_list_mat,
                                               load_spects=True)
        vds_json_str = vds.to_json(json_fname=None)
        vds_json = json.loads(vds_json_str)

        self.assertTrue(type(vds_json == dict))
        self.assertTrue('voc_list' in vds_json)
        voc_list = vds_json['voc_list']
        self.assertTrue(type(voc_list) == list)
        for voc in voc_list:
            for key in ['annot', 'duration', 'spect_path', 'metaspect', 'audio', 'audio_path']:
                self.assertTrue(key in voc)

        vds_from_json = Dataset.from_json(json_str=vds_json_str)
        self.assertTrue(type(vds_from_json) == Dataset)
        self.assertTrue(hasattr(vds, 'voc_list'))
        self.assertTrue(
            all([type(voc) == Vocalization for voc in vds.voc_list])
        )

    def test_Dataset_load_spects_mat(self):
        vds = vak.dataset.dataframe.from_files(spect_format='mat',
                                               spect_dir=self.spect_dir_mat,
                                               annot_list=self.annot_list_mat,
                                               load_spects=False)
        self.assertTrue(
            all([voc.metaspect is None for voc in vds.voc_list])
        )

        # check whether order of spect paths changes because we convert voc_list
        # to a dask bag to parallelize loading and order is usually preserved but this
        # is not guaranteed
        spect_paths_before = [voc.spect_path for voc in vds.voc_list]
        vds = vds.load_spects()
        self.assertTrue(
            all([type(voc.metaspect) == MetaSpect for voc in vds.voc_list])
        )
        # test that loaded metaspects have all attributes we expect them to have
        for voc in vds.voc_list:
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'spect', 'audio_path']:
                self.assertTrue(hasattr(voc.metaspect, attr))
                if attr in ['freq_bins', 'time_bins', 'spect']:
                    self.assertTrue(type(getattr(voc.metaspect, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(voc.metaspect, attr)) in (float, np.float16, np.float32, np.float64))
                elif attr == 'audio_path':
                    # check that we default to None when .mat file doesn't have 'audio_path' defined
                    self.assertTrue(getattr(voc.metaspect, attr) is None)

        spect_paths_after = [voc.spect_path for voc in vds.voc_list]
        for before, after in zip(spect_paths_before, spect_paths_after):
            self.assertTrue(
                before == after
            )

    def test_Dataset_load_spects_npz(self):
        vds = vak.dataset.dataframe.from_files(spect_format='npz',
                                               spect_dir=self.spect_dir_npz,
                                               annot_list=self.annot_list_cbin,
                                               load_spects=False)
        self.assertTrue(
            all([voc.metaspect is None for voc in vds.voc_list])
        )

        # check whether order of spect paths changes because we convert voc_list
        # to a dask bag to parallelize loading and order is usually preserved but this
        # is not guaranteed
        spect_paths_before = [voc.spect_path for voc in vds.voc_list]
        vds = vds.load_spects()
        self.assertTrue(
            all([type(voc.metaspect) == MetaSpect for voc in vds.voc_list])
        )
        # test that loaded metaspects have all attributes we expect them to have
        for voc in vds.voc_list:
            for attr in ['freq_bins', 'time_bins', 'timebin_dur', 'spect', 'audio_path']:
                self.assertTrue(hasattr(voc.metaspect, attr))
                if attr in ['freq_bins', 'time_bins', 'spect']:
                    self.assertTrue(type(getattr(voc.metaspect, attr)) == np.ndarray)
                elif attr == 'timebin_dur':
                    self.assertTrue(type(getattr(voc.metaspect, attr)) in (float, np.float16, np.float32, np.float64))
                elif attr == 'audio_path':
                    self.assertTrue(type(getattr(voc.metaspect, attr)) is str)

        spect_paths_after = [voc.spect_path for voc in vds.voc_list]
        for before, after in zip(spect_paths_before, spect_paths_after):
            self.assertTrue(
                before == after
            )

    def test_Dataset_clear_spects(self):
        vds = vak.dataset.dataframe.from_files(spect_format='mat',
                                               spect_dir=self.spect_dir_mat,
                                               annot_list=self.annot_list_mat,
                                               load_spects=True)
        self.assertTrue(
            all([type(voc.metaspect) == MetaSpect for voc in vds.voc_list])
        )

        vds = vds.clear_spects()
        self.assertTrue(
            all([voc.metaspect is None for voc in vds.voc_list])
        )

    def test_Dataset_move_spects_npz(self):
        vds = vak.dataset.dataframe.from_files(spect_format='npz',
                                               spect_dir=self.spect_dir_npz,
                                               annot_list=self.annot_list_cbin,
                                               load_spects=False)
        tmp_dir = tempfile.mkdtemp()
        dst = os.path.join(tmp_dir, self.spect_dir_npz.name)
        tmp_spects_dir = shutil.copytree(src=self.spect_dir_npz,
                                         dst=dst)

        vds = vds.move_spects(new_root=tmp_dir)
        self.assertTrue(
            all([tmp_spects_dir in voc.spect_path for voc in vds.voc_list])
        )

    def test_Dataset_move_spects_mat(self):
        vds = vak.dataset.dataframe.from_files(spect_format='mat',
                                               spect_dir=self.spect_dir_mat,
                                               annot_list=self.annot_list_mat,
                                               load_spects=True)

        tmp_dir = tempfile.mkdtemp()
        dst = os.path.join(tmp_dir, self.spect_dir_mat.name)
        tmp_spects_dir = shutil.copytree(src=self.spect_dir_mat,
                                         dst=dst)

        vds = vds.move_spects(new_root=tmp_dir, spect_dir_str='spect')
        self.assertTrue(
            all([tmp_spects_dir in voc.spect_path for voc in vds.voc_list])
        )

    def test_Dataset_are_spects_loaded(self):
        vds = vak.dataset.dataframe.from_files(spect_format='mat',
                                               spect_dir=self.spect_dir_mat,
                                               annot_list=self.annot_list_mat,
                                               load_spects=False)
        self.assertTrue(
            vds.are_spects_loaded() is False
        )

        vds = vds.load_spects()
        self.assertTrue(
            vds.are_spects_loaded() is True
        )

    def test_Dataset_spects_list(self):
        vds = vak.dataset.dataframe.from_files(spect_format='mat',
                                               spect_dir=self.spect_dir_mat,
                                               annot_list=self.annot_list_mat,
                                               load_spects=True)
        spects_list = vds.spects_list()
        self.assertTrue(
            type(spects_list == list)
        )
        self.assertTrue(
            all([type(spect) == np.ndarray for spect in spects_list])
        )
        self.assertTrue(
            len(spects_list) == len(vds.voc_list)
        )

    def test_Dataset_labels_list(self):
        vds = vak.dataset.dataframe.from_files(spect_format='mat',
                                               spect_dir=self.spect_dir_mat,
                                               annot_list=self.annot_list_mat,
                                               load_spects=False)
        labels_list = vds.labels_list()
        self.assertTrue(
            type(labels_list == list)
        )
        self.assertTrue(
            all([type(labels) == np.ndarray for labels in labels_list])
        )
        self.assertTrue(
            len(labels_list) == len(vds.voc_list)
        )

    def test_Dataset_lbl_tb_list(self):
        vds = vak.dataset.dataframe.from_files(labelset=self.labelset_mat,
                                               spect_format='mat',
                                               spect_dir=self.spect_dir_mat,
                                               annot_list=self.annot_list_mat,
                                               load_spects=True)
        lbl_tb_list = vds.lbl_tb_list()
        self.assertTrue(
            type(lbl_tb_list == list)
        )
        self.assertTrue(
            all([type(lbl_tb) == np.ndarray for lbl_tb in lbl_tb_list])
        )
        self.assertTrue(
            len(lbl_tb_list) == len(vds.voc_list)
        )


if __name__ == '__main__':
    unittest.main()
