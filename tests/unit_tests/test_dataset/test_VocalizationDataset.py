import os
import unittest
from glob import glob
import json

import numpy as np
from scipy.io import loadmat
import crowsetta

import vak.dataset.spect
import vak.dataset.annot
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


class TestVocalizationDataset(unittest.TestCase):
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

    def test_VocalizationDataset_init(self):
        voc_list = []
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
            for key in ['annot', 'duration', 'spect_path', 'metaspect', 'audio', 'audio_path']:
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
            all([voc.metaspect is None for voc in vds.voc_list])
        )

        # check whether order of spect paths changes because we convert voc_list
        # to a dask bag to parallelize loading and order is usually preserved but this
        # is not guaranteed
        spect_paths_before = [voc.spect_path for voc in vds.voc_list]
        vds.load_spects()
        self.assertTrue(
            all([type(voc.metaspect) == MetaSpect for voc in vds.voc_list])
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
            all([type(voc.metaspect) == MetaSpect for voc in vds.voc_list])
        )

        vds.clear_spects()
        self.assertTrue(
            all([voc.metaspect is None for voc in vds.voc_list])
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
        self.assertTrue(
            len(spects_list) == len(vds.voc_list)
        )

    def test_VocalizationDataset_labels_list(self):
        vds = vak.dataset.spect.from_files(spect_format='mat',
                                           spect_dir=self.spect_dir_mat,
                                           annot_list=self.annot_list,
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

    def test_VocalizationDataset_lbl_tb_list(self):
        vds = vak.dataset.spect.from_files(labelset=self.labelset_mat,
                                           spect_format='mat',
                                           spect_dir=self.spect_dir_mat,
                                           annot_list=self.annot_list,
                                           load_spects=True)
        labelmap = vak.utils.labels.to_map(vds.labelset)
        vds.labelmap = labelmap
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
