"""Tests for vak.prep.audio_dataset module."""
from pathlib import Path

import pandas as pd
import pytest

import vak.common.annotation
import vak.common.constants
import vak.prep.spectrogram_dataset.prep
import vak.prep.spectrogram_dataset.spect_helper


def returned_dataframe_matches_expected(
    df_returned,
    data_dir,
    labelset,
    annot_format,
    audio_format,
    spect_format,
    spect_output_dir=None,
    annot_file=None,
    expected_audio_paths=None,
    not_expected_audio_paths=None,
    spect_file_ext=".spect.npz",
    expected_spect_paths=None,
    not_expected_spect_paths=None,
):
    """tests that dataframe returned by ``vak.prep.spectrogram_dataset.prep.prep_spectrogram_dataset``
    matches expected dataframe."""
    assert isinstance(df_returned, pd.DataFrame)

    assert df_returned.columns.values.tolist() == vak.prep.spectrogram_dataset.spect_helper.DF_COLUMNS

    annot_format_from_df = df_returned.annot_format.unique()
    assert len(annot_format_from_df) == 1
    annot_format_from_df = annot_format_from_df[0]

    if annot_format is not None:
        # test format from dataframe is specified format
        assert annot_format_from_df == annot_format
        annot_list = vak.common.annotation.from_df(df_returned)
    else:
        # if no annotation format specified, check that `annot_format` is `none`
        assert annot_format_from_df == vak.common.constants.NO_ANNOTATION_FORMAT
        annot_list = None

    if annot_file:
        assert all(
            [Path(annot_path) == annot_file for annot_path in df_returned["annot_path"]]
        )

    if labelset:
        # test that filtering vocalizations by labelset worked
        assert annot_list is not None, "labelset specified but annot_list was None"
        for annot in annot_list:
            labels = annot.seq.labels
            labelset_from_labels = set(labels)
            # should be true that set of labels from each annotation is a subset of labelset
            assert labelset_from_labels.issubset(set(labelset))

    audio_files_from_df = [Path(audio_path) for audio_path in df_returned.audio_path]
    spect_paths_from_df = [Path(spect_path) for spect_path in df_returned.spect_path]

    # --- which assertions to run depends on whether we made the dataframe
    # from audio files or spect files ----
    if audio_format:  # implies that --> we made the dataframe from audio files

        # test that all audio files came from data_dir
        for audio_file_from_df in audio_files_from_df:
            assert Path(audio_file_from_df).parent == data_dir

        # test that each audio file has a corresponding spect file in `spect_path` column
        spect_file_names = [spect_path.name for spect_path in spect_paths_from_df]
        expected_spect_files = [
            source_audio_file.name + spect_file_ext
            for source_audio_file in expected_audio_paths
        ]
        assert all(
            [
                expected_spect_file in spect_file_names
                for expected_spect_file in expected_spect_files
            ]
        )

        # if there are audio files we expect to **not** be in audio_path
        # -- because the associated annotations have labels not in labelset --
        # then test those files are **not** in spect_path
        if not_expected_audio_paths is not None:
            not_expected_spect_files = [
                source_audio_file.name + spect_file_ext
                for source_audio_file in not_expected_audio_paths
            ]
            assert all(
                [
                    not_expected_spect_file not in spect_file_names
                    for not_expected_spect_file in not_expected_spect_files
                ]
            )

        # test that all the generated spectrogram files are in a
        # newly-created directory inside spect_output_dir
        assert all(
            [
                spect_path.parents[1] == spect_output_dir
                for spect_path in spect_paths_from_df
            ]
        )

    elif spect_format:  # implies that --> we made the dataframe from audio files
        for expected_spect_path in list(expected_spect_paths):
            assert expected_spect_path in spect_paths_from_df
            spect_paths_from_df.remove(expected_spect_path)

        # test that **only** expected paths were in DataFrame
        if not_expected_spect_paths is not None:
            for not_expected_spect_path in not_expected_spect_paths:
                assert not_expected_spect_path not in spect_paths_from_df

        # test that **only** expected paths were in DataFrame
        # spect_paths_from_df should be empty after popping off all the expected paths
        assert (
            len(spect_paths_from_df) == 0
        )  # yes I know this isn't "Pythonic". It's readable, go away.

    return True  # all asserts passed


# @pytest.mark.parametrize(
#     'data_dir, labelset,'
# )
def test_prep_audio_dataset(
    # audio_dir_cbin,
    # default_spect_params,
    # labelset_notmat,
    # audio_list_cbin_all_labels_in_labelset,
    # audio_list_cbin_labels_not_in_labelset,
    # spect_list_npz_all_labels_in_labelset,
    # spect_list_npz_labels_not_in_labelset,
    # tmp_path,
):
    """test that ``vak.prep.spectrogram_dataset.prep.prep_spectrogram_dataset`` works
    when we point it at directory of .cbin audio files
    and specify an annotation format"""
    assert False
    #
    # dataset_df = vak.prep.audio_dataset.prep_audio_dataset(
    #     data_dir=audio_dir_cbin,
    #     labelset=labelset_notmat,
    #     annot_format="notmat",
    #     audio_format="cbin",
    #     annot_file=None,
    # )
    #
    # assert returned_dataframe_matches_expected(
    #     dataset_df,
    #     data_dir=audio_dir_cbin,
    #     labelset=labelset_notmat,
    #     annot_format="notmat",
    #     audio_format="cbin",
    #     spect_format=None,
    #     spect_output_dir=tmp_path,
    #     annot_file=None,
    #     expected_audio_paths=audio_list_cbin_all_labels_in_labelset,
    #     not_expected_audio_paths=audio_list_cbin_labels_not_in_labelset,
    #     expected_spect_paths=spect_list_npz_all_labels_in_labelset,
    #     not_expected_spect_paths=spect_list_npz_labels_not_in_labelset,
    # )
    #
    # assert spect_files_have_correct_keys(vak_df, default_spect_params)
