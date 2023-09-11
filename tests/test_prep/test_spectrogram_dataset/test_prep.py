"""tests for vak.prep.spectrogram_dataset module"""
import pathlib

import pandas as pd
import pytest

import vak.common.annotation
import vak.common.constants
import vak.prep.spectrogram_dataset.prep
import vak.prep.spectrogram_dataset.spect_helper

from ...fixtures.annot import ANNOT_FILE_YARDEN
from ...fixtures.audio import AUDIO_DIR_CBIN
from ...fixtures.spect import SPECT_DIR_MAT


def assert_returned_dataframe_matches_expected(
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
            [pathlib.Path(annot_path) == annot_file for annot_path in df_returned["annot_path"]]
        )

    if labelset:
        # test that filtering vocalizations by labelset worked
        assert annot_list is not None, "labelset specified but annot_list was None"
        for annot in annot_list:
            labels = annot.seq.labels
            labelset_from_labels = set(labels)
            # should be true that set of labels from each annotation is a subset of labelset
            assert labelset_from_labels.issubset(set(labelset))

    audio_paths_from_df = [pathlib.Path(audio_path) for audio_path in df_returned.audio_path]
    spect_paths_from_df = [pathlib.Path(spect_path) for spect_path in df_returned.spect_path]
    spect_file_names_from_df = [spect_path.name for spect_path in spect_paths_from_df]

    # which assertions to run depends on whether we made the dataframe
    # from audio files or spect files
    if audio_format:  # implies that --> we made the dataframe from audio files

        # test that all audio files came from data_dir
        for audio_path_from_df in audio_paths_from_df:
            assert audio_path_from_df.parent == data_dir

        # test that each audio file has a corresponding spect file in `spect_path` column
        expected_spect_files = [
            source_audio_file.name + spect_file_ext
            for source_audio_file in expected_audio_paths
        ]
        assert all(
            [
                expected_spect_file in spect_file_names_from_df
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
                    not_expected_spect_file not in spect_file_names_from_df
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

    elif spect_format:  # implies that --> we made the dataframe from spect files
        if spect_format == 'mat':
            expected_spect_file_names = [
                spect_path.name.replace('.mat', '.npz')
                for spect_path in expected_spect_paths
            ]
        else:
            expected_spect_file_names = [
                spect_path.name for spect_path in expected_spect_paths
            ]

        assert all(
            [expected_spect_file_name in spect_file_names_from_df
             for expected_spect_file_name in expected_spect_file_names]
        )

        # test that **only** expected paths were in DataFrame
        if not_expected_spect_paths is not None:
            if spect_format == 'mat':
                not_expected_spect_file_names = [
                    spect_path.name.replace('.mat', '.npz')
                    for spect_path in not_expected_spect_paths
                ]
            else:
                not_expected_spect_file_names = [
                    spect_path.name for spect_path in not_expected_spect_paths
                ]
            assert all(
                [not_expected_spect_file_name not in spect_file_names_from_df
                 for not_expected_spect_file_name in not_expected_spect_file_names]
            )


def assert_spect_files_have_correct_keys(df_returned,
                                  spect_params):
    spect_paths_from_df = [pathlib.Path(spect_path) for spect_path in df_returned.spect_path]
    for spect_path in spect_paths_from_df:
        spect_dict = vak.common.files.spect.load(spect_path)
        for key_type in ['freqbins_key', 'timebins_key', 'spect_key']:
            if key_type in spect_params:
                key = spect_params[key_type]
            else:
                # if we didn't pass in this key type, don't check for it
                # this is for `audio_path` which is not strictly required currently
                continue
            assert key in spect_dict


@pytest.mark.parametrize(
    'data_dir, audio_format, spect_format, annot_format, labelset, annot_file',
    [
        (AUDIO_DIR_CBIN, "cbin", None, "notmat", True, None),
        (AUDIO_DIR_CBIN, "cbin", None, "notmat", False, None),
        (AUDIO_DIR_CBIN, "cbin", None, None, False, None),
        (SPECT_DIR_MAT, None, "mat", "yarden", True, ANNOT_FILE_YARDEN),
        (SPECT_DIR_MAT, None, "mat", "yarden", False, ANNOT_FILE_YARDEN),
        (SPECT_DIR_MAT, None, "mat", None, False, None),
    ]
)
def test_prep_spectrogram_dataset(
    data_dir,
    audio_format,
    spect_format,
    annot_format,
    labelset,
    annot_file,
    tmp_path,
    default_spect_params,
    specific_audio_list,
    specific_spect_list,
    specific_labelset,
):
    """Test that ``vak.prep.spectrogram_dataset.prep.prep_spectrogram_dataset`` works
    when we point it at directory of .cbin audio files
    and  **do not** specify an annotation format"""
    if labelset:
        labelset = specific_labelset(annot_format)
    else:
        labelset = None

    dataset_df = vak.prep.spectrogram_dataset.prep.prep_spectrogram_dataset(
        data_dir=data_dir,
        annot_format=annot_format,
        labelset=labelset,
        annot_file=annot_file,
        audio_format=audio_format,
        spect_output_dir=tmp_path,
        spect_format=spect_format,
        spect_params=default_spect_params,
    )

    if labelset and audio_format:
        expected_audio_paths = specific_audio_list(audio_format, "all_labels_in_labelset")
        not_expected_audio_paths = specific_audio_list(audio_format, "labels_not_in_labelset")
        expected_spect_paths = None
        not_expected_spect_paths = None
    elif labelset is None and audio_format:
        expected_audio_paths = specific_audio_list(audio_format)
        not_expected_audio_paths = None
        expected_spect_paths = None
        not_expected_spect_paths = None
    elif labelset and spect_format:
        expected_audio_paths = None
        not_expected_audio_paths = None
        expected_spect_paths = specific_spect_list(
            spect_format, "all_labels_in_labelset"
        )
        not_expected_spect_paths = specific_spect_list(
            spect_format, "labels_not_in_labelset"
        )
    elif labelset is None and spect_format:
        expected_audio_paths = None
        not_expected_audio_paths = None
        expected_spect_paths = specific_spect_list(spect_format)
        not_expected_spect_paths = None

    assert_returned_dataframe_matches_expected(
        dataset_df,
        data_dir=data_dir,
        annot_format=annot_format,
        labelset=labelset,
        audio_format=audio_format,
        spect_format=spect_format,
        spect_output_dir=tmp_path,
        expected_audio_paths=expected_audio_paths,
        not_expected_audio_paths=not_expected_audio_paths,
        expected_spect_paths=expected_spect_paths,
        not_expected_spect_paths=not_expected_spect_paths,
        annot_file=annot_file,
    )

    assert_spect_files_have_correct_keys(dataset_df, default_spect_params)
