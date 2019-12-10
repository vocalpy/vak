import os
from pathlib import Path
import unittest
from glob import glob

import crowsetta

import vak.io.annotation
import vak.io.audio

HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')
SETUP_SCRIPTS_DIR = HERE.joinpath('..', '..', 'setup_scripts')


class TestAnnot(unittest.TestCase):

    def test_source_annot_map_wav_koumura(self):
        scribe = crowsetta.Transcriber(voc_format='koumura')
        koumura_dir = TEST_DATA_DIR.joinpath('koumura', 'Bird0')
        annot_xml = str(koumura_dir.joinpath('Annotation.xml'))
        wavpath = koumura_dir.joinpath('Wave')
        annot_list = scribe.to_seq(file=annot_xml, wavpath=str(wavpath))
        audio_files = wavpath.glob('*.wav')
        audio_files = [str(path) for path in audio_files]
        source_annot_map = vak.io.annotation.source_annot_map(source_files=audio_files,
                                                              annot_list=annot_list)

        for source, annot in list(source_annot_map.items()):
            self.assertTrue(source in audio_files)
            self.assertTrue(annot in annot_list)
            source_annot_map.pop(source)

        # if every source file got mapped to an annot, and we mapped all of them,
        # then dictionary should be empty after loop
        self.assertTrue(source_annot_map == {})

    def test_source_annot_map_cbin_notmat(self):
        scribe = crowsetta.Transcriber(voc_format='notmat')
        cbin_dir = TEST_DATA_DIR.joinpath('cbins', 'gy6or6', '032312')
        notmats = cbin_dir.glob('*.not.mat')
        notmats = [str(path) for path in notmats]
        annot_list = scribe.to_seq(file=notmats)

        audio_files = cbin_dir.glob('*.cbin')
        audio_files = [str(path) for path in audio_files]
        source_annot_map = vak.io.annotation.source_annot_map(source_files=audio_files,
                                                              annot_list=annot_list)

        for source, annot in list(source_annot_map.items()):
            self.assertTrue(source in audio_files)
            self.assertTrue(annot in annot_list)
            source_annot_map.pop(source)

        # if every source file got mapped to an annot, and we mapped all of them,
        # then dictionary should be empty after loop
        self.assertTrue(source_annot_map == {})

    def test_source_annot_map_cbin_yarden(self):
        scribe = crowsetta.Transcriber(voc_format='yarden')
        mat_dir = TEST_DATA_DIR.joinpath('mat', 'llb3')
        annot_file = str(mat_dir.joinpath('llb3_annot_subset.mat'))
        annot_list = scribe.to_seq(file=annot_file)

        spect_files = mat_dir.joinpath('spect').glob('*.mat')
        spect_files = [str(path) for path in spect_files]

        source_annot_map = vak.io.annotation.source_annot_map(source_files=spect_files,
                                                              annot_list=annot_list)

        for source, annot in list(source_annot_map.items()):
            self.assertTrue(source in spect_files)
            self.assertTrue(annot in annot_list)
            source_annot_map.pop(source)

        # if every source file got mapped to an annot, and we mapped all of them,
        # then dictionary should be empty after loop
        self.assertTrue(source_annot_map == {})


    def test_files_from_dir(self):
        notmat_dir = os.path.join(TEST_DATA_DIR, 'cbins', 'gy6or6', '032312')
        annot_files = vak.io.annotation.files_from_dir(notmat_dir, annot_format='notmat')

        notmat_files = glob(os.path.join(notmat_dir, '*.not.mat'))
        self.assertTrue(
            sorted(annot_files) == sorted(notmat_files)
        )


if __name__ == '__main__':
    unittest.main()
