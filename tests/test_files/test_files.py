"""tests for vak.files.files.files module"""
import pytest

import vak.files.files


def test_find_fname():
    fname = "llb3_0003_2018_04_23_14_18_54.wav.mat"
    ext = "wav"
    out = vak.files.files.find_fname(fname, ext)
    assert out == "llb3_0003_2018_04_23_14_18_54.wav"


def test_files_from_dir_with_mat(spect_dir_mat, spect_list_mat):
    files = vak.files.files.from_dir(spect_dir_mat, "mat")
    # files.from_dir returns str not Path, need to convert fixture
    spect_list_mat = [str(spect_path) for spect_path in spect_list_mat]
    assert sorted(spect_list_mat) == sorted(files)


def test_files_from_dir_with_cbin(audio_dir_cbin, audio_list_cbin):
    files = vak.files.files.from_dir(audio_dir_cbin, "cbin")
    # files.from_dir returns str not Path, need to convert fixture
    audio_list_cbin = [str(audio_path) for audio_path in audio_list_cbin]
    assert sorted(audio_list_cbin) == sorted(files)


@pytest.mark.parametrize(
    ("dir_path", "ext"),
    [
        ("./tests/data_for_tests/source/audio_wav_annot_textgrid/AGBk/", "WAV"),
        ("./tests/data_for_tests/source/audio_wav_annot_birdsongrec/Bird0/Wave", "wav"),
    ],
)
def test_from_dir_is_case_insensitive(dir_path, ext):
    files = vak.files.files.from_dir(dir_path, ext)
    assert len(files) > 0
    assert all([str(file).endswith(ext) for file in files])


@pytest.mark.parametrize(
    ("dir_path", "ext"),
    [
        ("./tests/data_for_tests/source/audio_wav_annot_textgrid/", "WAV"),
        ("./tests/data_for_tests/source/audio_wav_annot_birdsongrec/Bird0", "wav"),
    ],
)
def test_from_dir_searches_child_dir(dir_path, ext):
    files = vak.files.files.from_dir(dir_path, ext)
    assert len(files) > 0
    assert all([str(file).endswith(ext) for file in files])
