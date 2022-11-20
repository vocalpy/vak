"""tests for vak.files.files.files module"""
import itertools

import pytest

import vak.files.files

from ..fixtures.spect import (
    SPECT_LIST_MAT,
    SPECT_LIST_NPZ,
)

SPECT_FNAME_LIST_MAT = [
    spect_path.name for spect_path in SPECT_LIST_MAT
] + [
    # duplicate list but replace underscores in filenames with spaces to test handling spaces
    spect_path.name.replace('_', ' ') for spect_path in SPECT_LIST_MAT
]

SPECT_FNAME_LIST_NPZ = [
    spect_path.name for spect_path in SPECT_LIST_NPZ
] + [
    spect_path.name.replace('_', ' ') for spect_path in SPECT_LIST_NPZ
]

SPECT_FNAME_LIST_MAT_WITH_EXT = list(zip(
    SPECT_FNAME_LIST_MAT,
    itertools.repeat('.wav'),
    itertools.repeat('.mat'),
))

SPECT_FNAME_LIST_NPZ_WITH_EXT = list(zip(
    SPECT_FNAME_LIST_NPZ,
    itertools.repeat('.cbin'),
    itertools.repeat('.spect.npz'),
))

TEST_FIND_FNAME_PARAMETRIZE = SPECT_FNAME_LIST_MAT_WITH_EXT + SPECT_FNAME_LIST_NPZ_WITH_EXT


@pytest.mark.parametrize(
    'fname, find_ext, spect_ext',
    TEST_FIND_FNAME_PARAMETRIZE
)
def test_find_fname(fname, find_ext, spect_ext):
    """Test ``vak.files.files.find_fname`` works as expected."""
    expected = fname.replace(spect_ext, '')

    out = vak.files.files.find_fname(fname, find_ext)

    assert fname.startswith(out)
    assert out == expected


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
