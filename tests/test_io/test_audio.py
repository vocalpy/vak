"""tests for ``vak.io.audio`` module"""
from pathlib import Path

import numpy as np
import pytest

import vak.io.audio


def expected_spect_files_returned(
    spect_files_returned,
    source_audio_files_expected,
    source_audio_files_not_expected=None,
    spect_file_ext=".spect.npz",
):
    """test that ``spect_files`` returned by ``vak.io.audio.to_spect`` matches
    an expected list of spectrogram files

    Parameters
    ----------
    spect_files_returned : list
        of str, full paths to .npz array files containing spectrograms
    source_audio_files_expected : list
        of Path, source audio files expected to be used for spectrograms
    source_audio_files_not_expected : list
        of Path, source audio files expected to be used for spectrograms
    spect_file_ext : str
        extension given to array files containing spectrograms.
        Default is `.spect.npz`
    """
    assert type(spect_files_returned) == list

    spect_files_returned = [Path(spect_file) for spect_file in spect_files_returned]
    assert all([spect_file.exists() for spect_file in spect_files_returned])

    for spect_file in spect_files_returned:
        spect_dict = np.load(spect_file)
        for key in ["s", "f", "t"]:
            assert key in spect_dict
            assert type(spect_dict[key]) == np.ndarray

    # remove path so we can just compare file names
    spect_files_returned = [spect_file.name for spect_file in spect_files_returned]
    expected_spect_files = [
        source_audio_file.name + spect_file_ext
        for source_audio_file in source_audio_files_expected
    ]
    assert all(
        [
            expected_spect_file in spect_files_returned
            for expected_spect_file in expected_spect_files
        ]
    )

    if source_audio_files_not_expected is not None:
        not_expected_spect_files = [
            source_audio_file.name + spect_file_ext
            for source_audio_file in source_audio_files_not_expected
        ]
        assert all(
            [
                not_expected_spect_file not in spect_files_returned
                for not_expected_spect_file in not_expected_spect_files
            ]
        )

    return True  # if all asserts were True


def test_to_spect_audio_dir_annot_cbin_with_labelset(
    default_spect_params,
    tmp_path,
    audio_dir_cbin,
    annot_list_notmat,
    labelset_notmat,
    audio_list_cbin_all_labels_in_labelset,
    audio_list_cbin_labels_not_in_labelset,
):
    """test that ``vak.io.audio.to_spect`` works
    when we point it at directory of .cbin files + give it list of annotations"""
    spect_files = vak.io.audio.to_spect(
        audio_format="cbin",
        spect_params=default_spect_params,
        output_dir=tmp_path,
        audio_dir=audio_dir_cbin,
        audio_files=None,
        annot_list=annot_list_notmat,
        audio_annot_map=None,
        labelset=labelset_notmat,
    )
    assert expected_spect_files_returned(
        spect_files,
        audio_list_cbin_all_labels_in_labelset,
        audio_list_cbin_labels_not_in_labelset,
    )


def test_audio_dir_annot_cbin_no_labelset(
    default_spect_params,
    tmp_path,
    audio_dir_cbin,
    audio_list_cbin,
    annot_list_notmat,
):
    """test that ``vak.io.audio.to_spect`` works
    when we point it at directory of .cbin files + give it list of annotations
    but do not specify a set of labels"""
    spect_files = vak.io.audio.to_spect(
        audio_format="cbin",
        spect_params=default_spect_params,
        output_dir=tmp_path,
        audio_dir=audio_dir_cbin,
        audio_files=None,
        annot_list=annot_list_notmat,
        audio_annot_map=None,
        labelset=None,
    )
    assert expected_spect_files_returned(spect_files, audio_list_cbin)


def test_audio_dir_without_annot(
    default_spect_params,
    tmp_path,
    audio_dir_cbin,
    audio_list_cbin,
):
    """test that ``vak.io.audio.to_spect`` works
    when we point it at directory of .cbin files
    + give it list of annotations
    but do not specify a set of labels"""
    # make sure we can make a spectrograms from audio files without annotations,
    # e.g. if we're going to predict the annotations using the spectrograms
    spect_files = vak.io.audio.to_spect(
        audio_format="cbin",
        spect_params=default_spect_params,
        output_dir=tmp_path,
        audio_dir=audio_dir_cbin,
        audio_files=None,
        annot_list=None,
        audio_annot_map=None,
        labelset=None,
    )
    assert expected_spect_files_returned(spect_files, audio_list_cbin)


def test_audio_files_cbin_annot_notmat(
    default_spect_params,
    tmp_path,
    audio_list_cbin,
    annot_list_notmat,
    labelset_notmat,
    audio_list_cbin_all_labels_in_labelset,
    audio_list_cbin_labels_not_in_labelset,
):
    """test that ``vak.io.audio.to_spect`` works
    when we give it a list of .cbin files + give it list of annotations"""
    spect_files = vak.io.audio.to_spect(
        audio_format="cbin",
        spect_params=default_spect_params,
        output_dir=tmp_path,
        audio_dir=None,
        audio_files=audio_list_cbin,
        annot_list=annot_list_notmat,
        audio_annot_map=None,
        labelset=labelset_notmat,
    )
    assert expected_spect_files_returned(
        spect_files,
        audio_list_cbin_all_labels_in_labelset,
        audio_list_cbin_labels_not_in_labelset,
    )


def test_audio_files_annot_cbin_no_labelset(
    default_spect_params,
    tmp_path,
    audio_list_cbin,
    annot_list_notmat,
):
    """test that ``vak.io.audio.to_spect`` works
    when we give it a list of .cbin files
    + give it list of annotations
    but do not specify a set of labels"""
    spect_files = vak.io.audio.to_spect(
        audio_format="cbin",
        spect_params=default_spect_params,
        output_dir=tmp_path,
        audio_dir=None,
        audio_files=audio_list_cbin,
        annot_list=annot_list_notmat,
        audio_annot_map=None,
        labelset=None,
    )
    assert expected_spect_files_returned(spect_files, audio_list_cbin)


def test_audio_annot_map_cbin(
    default_spect_params,
    tmp_path,
    audio_list_cbin,
    annot_list_notmat,
    labelset_notmat,
    audio_list_cbin_all_labels_in_labelset,
    audio_list_cbin_labels_not_in_labelset,
):
    """test that ``vak.io.audio.to_spect`` works
    when we give it a dict that maps audio files to annotations"""
    audio_annot_map = dict(zip(audio_list_cbin, annot_list_notmat))
    spect_files = vak.io.audio.to_spect(
        audio_format="cbin",
        spect_params=default_spect_params,
        output_dir=tmp_path,
        audio_dir=None,
        audio_files=None,
        annot_list=None,
        audio_annot_map=audio_annot_map,
        labelset=labelset_notmat,
    )
    assert expected_spect_files_returned(
        spect_files,
        audio_list_cbin_all_labels_in_labelset,
        audio_list_cbin_labels_not_in_labelset,
    )


def test_audio_annot_map_cbin_no_labelset(
    default_spect_params,
    tmp_path,
    audio_list_cbin,
    annot_list_notmat,
    labelset_notmat,
):
    """test that ``vak.io.audio.to_spect`` works
    when we give it a dict that maps audio files to annotations
    but do not give it a labelset to filter out files"""
    audio_annot_map = dict(zip(audio_list_cbin, annot_list_notmat))
    spect_files = vak.io.audio.to_spect(
        audio_format="cbin",
        spect_params=default_spect_params,
        output_dir=tmp_path,
        audio_dir=None,
        audio_files=None,
        annot_list=None,
        audio_annot_map=audio_annot_map,
        labelset=None,
    )
    assert expected_spect_files_returned(spect_files, audio_list_cbin)


def test_missing_inputs_raise(
    default_spect_params,
    tmp_path,
    audio_dir_cbin,
    audio_list_cbin,
    annot_list_notmat,
    labelset_notmat,
):
    """test that calling ``vak.io.audio.to_spect`` without one of:
    audio files, audio list, or audio files/annotations mapping
    raises a ValueError
    """
    with pytest.raises(ValueError):
        vak.io.audio.to_spect(
            audio_format="ape",
            spect_params=default_spect_params,
            output_dir=tmp_path,
            audio_dir=None,
            audio_files=None,
            annot_list=annot_list_notmat,
            audio_annot_map=None,
            labelset=labelset_notmat,
        )


def test_invalid_audio_format_raises(
    default_spect_params,
    tmp_path,
    audio_dir_cbin,
    audio_list_cbin,
    annot_list_notmat,
    labelset_notmat,
):
    """test that calling ``vak.io.audio.to_spect`` with an invalid audio format
    raises a ValueError"""
    with pytest.raises(ValueError):
        vak.io.audio.to_spect(
            audio_format="ape",
            spect_params=default_spect_params,
            output_dir=tmp_path,
            audio_dir=audio_dir_cbin,
            audio_files=None,
            annot_list=annot_list_notmat,
            audio_annot_map=None,
            labelset=labelset_notmat,
        )


def test_both_audio_dir_and_audio_files_raises(
    default_spect_params,
    tmp_path,
    audio_dir_cbin,
    audio_list_cbin,
    annot_list_notmat,
    labelset_notmat,
):
    """ "test that calling ``vak.io.audio.to_spect``
    with both audio_dir and audio_files
    raises a ValueError"""
    with pytest.raises(ValueError):
        vak.io.audio.to_spect(
            audio_format="cbin",
            spect_params=default_spect_params,
            output_dir=tmp_path,
            audio_dir=audio_dir_cbin,
            audio_files=audio_list_cbin,
            annot_list=annot_list_notmat,
            audio_annot_map=None,
            labelset=labelset_notmat,
        )


def test_both_audio_dir_and_audio_annot_map_raises(
    default_spect_params,
    tmp_path,
    audio_dir_cbin,
    audio_list_cbin,
    annot_list_notmat,
    labelset_notmat,
):
    """ "test that calling ``vak.io.audio.to_spect``
    with both audio_dir and audio_annot_map
    raises a ValueError"""
    audio_annot_map = dict(zip(audio_list_cbin, annot_list_notmat))
    with pytest.raises(ValueError):
        vak.io.audio.to_spect(
            audio_format="cbin",
            spect_params=default_spect_params,
            output_dir=tmp_path,
            audio_dir=audio_dir_cbin,
            audio_files=None,
            annot_list=None,
            audio_annot_map=audio_annot_map,
            labelset=labelset_notmat,
        )


def test_both_audio_list_and_audio_annot_map_raises(
    default_spect_params,
    tmp_path,
    audio_dir_cbin,
    audio_list_cbin,
    annot_list_notmat,
    labelset_notmat,
):
    """ "test that calling ``vak.io.audio.to_spect``
    with both audio_files and audio_annot_map
    raises a ValueError"""
    audio_annot_map = dict(zip(audio_list_cbin, annot_list_notmat))
    with pytest.raises(ValueError):
        vak.io.audio.to_spect(
            audio_format="cbin",
            spect_params=default_spect_params,
            output_dir=tmp_path,
            audio_dir=None,
            audio_files=audio_list_cbin,
            annot_list=None,
            audio_annot_map=audio_annot_map,
            labelset=labelset_notmat,
        )


def test_both_annot_list_and_audio_annot_map_raises(
    default_spect_params,
    tmp_path,
    audio_dir_cbin,
    audio_list_cbin,
    annot_list_notmat,
    labelset_notmat,
):
    """ "test that calling ``vak.io.audio.to_spect``
    with both audio_annot_map and annot_list
    raises a ValueError"""
    audio_annot_map = dict(zip(audio_list_cbin, annot_list_notmat))
    with pytest.raises(ValueError):
        vak.io.audio.to_spect(
            audio_format="cbin",
            spect_params=default_spect_params,
            output_dir=tmp_path,
            audio_dir=None,
            audio_files=None,
            annot_list=annot_list_notmat,
            audio_annot_map=audio_annot_map,
            labelset=labelset_notmat,
        )


@pytest.mark.parametrize(
    'dask_bag_kwargs',
    [
        None,
        dict(npartitions=20),
    ]
)
def test_dask_bag_kwargs(
    dask_bag_kwargs,
    default_spect_params,
    tmp_path,
    audio_dir_cbin,
    annot_list_notmat,
    labelset_notmat,
    audio_list_cbin_all_labels_in_labelset,
    audio_list_cbin_labels_not_in_labelset,

):
    """Test the ``dask_bag_kwargs`` parameter.

    This is a smoke test, it does not carefully test different parameters.
    It's the test above ``test_to_spect_audio_dir_annot_cbin_with_labelset``
    with the parameter ``dask_bag_kwargs`` added."""
    spect_files = vak.io.audio.to_spect(
        audio_format="cbin",
        spect_params=default_spect_params,
        output_dir=tmp_path,
        audio_dir=audio_dir_cbin,
        audio_files=None,
        annot_list=annot_list_notmat,
        audio_annot_map=None,
        labelset=labelset_notmat,
        dask_bag_kwargs=dask_bag_kwargs
    )
    assert expected_spect_files_returned(
        spect_files,
        audio_list_cbin_all_labels_in_labelset,
        audio_list_cbin_labels_not_in_labelset,
    )
