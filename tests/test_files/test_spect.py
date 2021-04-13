"""tests for vak.files.spect module"""
from pathlib import Path

import pytest

import vak.files
from vak.constants import VALID_AUDIO_FORMATS


@pytest.mark.parametrize(
    "spect_format",
    [
        "mat",
        "npz",
    ],
)
def test_find_audio_fname_with_mat(spect_format, specific_spect_list):
    """test ```vak.files.spect.find_audio_fname`` works when we give it a list of """
    spect_list = specific_spect_list(spect_format)
    audio_fnames = [
        vak.files.spect.find_audio_fname(spect_path) for spect_path in spect_list
    ]
    for spect_path, audio_fname in zip(spect_list, audio_fnames):
        # make sure we gout out a filename that was actually in spect_path
        assert spect_path.name.startswith(audio_fname)
        # make sure it's some valid audio format
        assert Path(audio_fname).suffix.replace(".", "") in VALID_AUDIO_FORMATS
