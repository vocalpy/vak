"""tests for vak.files.spect module"""
import itertools
import pathlib

import pytest

import vak.files
from vak.constants import VALID_AUDIO_FORMATS


from ..fixtures.spect import (
    SPECT_LIST_MAT,
    SPECT_LIST_NPZ,
)


SPECT_LIST_MAT = SPECT_LIST_MAT + [
    # duplicate list but replace underscores in filenames with spaces to test handling spaces
    spect_path.parent / spect_path.name.replace('_', ' ')
    for spect_path in SPECT_LIST_MAT
] + [
    # duplicate list but replace underscores in entire path with spaces to test handling spaces
    pathlib.Path(str(spect_path).replace('_', ' '))
    for spect_path in SPECT_LIST_MAT
]
# test function with string paths not just pathlib.Path instances
SPECT_LIST_MAT = SPECT_LIST_MAT + [str(path) for path in SPECT_LIST_MAT]

SPECT_LIST_NPZ = SPECT_LIST_NPZ + [
    # duplicate list but replace underscores in filenames with spaces to test handling spaces
    spect_path.parent / spect_path.name.replace('_', ' ')
    for spect_path in SPECT_LIST_NPZ
] + [
    # duplicate list but replace underscores in entire path with spaces to test handling spaces
    pathlib.Path(str(spect_path).replace('_', ' '))
    for spect_path in SPECT_LIST_NPZ
]
# test function with string paths not just pathlib.Path instances
SPECT_LIST_NPZ = SPECT_LIST_NPZ + [str(path) for path in SPECT_LIST_NPZ]

SPECT_LIST_MAT_WITH_EXT = list(zip(
    SPECT_LIST_MAT,
    itertools.repeat('.wav'),
    itertools.repeat('.mat'),
)) + list(zip(
    SPECT_LIST_MAT,
    # also test with case where we don't specify audio extension
    itertools.repeat(None),
    itertools.repeat('.mat'),
))

SPECT_LIST_NPZ_WITH_EXT = list(zip(
    SPECT_LIST_NPZ,
    itertools.repeat('.cbin'),
    itertools.repeat('.spect.npz'),
)) + list(zip(
    SPECT_LIST_NPZ,
    itertools.repeat(None),
    itertools.repeat('.spect.npz'),
))

TEST_FIND_AUDIO_FNAME_PARAMETRIZE = SPECT_LIST_MAT_WITH_EXT + SPECT_LIST_NPZ_WITH_EXT


@pytest.mark.parametrize(
    "spect_path, audio_ext, spect_ext",
    TEST_FIND_AUDIO_FNAME_PARAMETRIZE
)
def test_find_audio_fname_with_mat(spect_path, audio_ext, spect_ext):
    """test ```vak.files.spect.find_audio_fname`` works when we give it a list of """
    expected = pathlib.Path(spect_path).name.replace(spect_ext, '')

    out = vak.files.spect.find_audio_fname(spect_path, audio_ext)

    # make sure we gout out a filename that was actually in spect_path
    assert pathlib.Path(spect_path).name.startswith(out)
    # make sure it's some valid audio format
    assert pathlib.Path(out).suffix.replace(".", "") in VALID_AUDIO_FORMATS
    assert out == expected
