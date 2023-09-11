"""Tests for vak.prep.audio_dataset module."""
from pathlib import Path

import pandas as pd
import pytest

from ..fixtures.audio import (
    AUDIO_DIR_CBIN, AUDIO_LIST_CBIN_ALL_LABELS_IN_LABELSET, AUDIO_LIST_CBIN_LABELS_NOT_IN_LABELSET
)
from ..fixtures.annot import LABELSET_NOTMAT

import vak.common.annotation
import vak.common.constants
import vak.prep.spectrogram_dataset.prep
import vak.prep.spectrogram_dataset.spect_helper


def returned_dataframe_matches_expected(
    df_returned,
    data_dir,
    labelset,
    audio_format,
    annot_format=None,
    annot_file=None,
    expected_audio_paths=None,
    not_expected_audio_paths=None,
):
    """tests that dataframe returned by ``vak.prep.audio_dataset.prep.prep_spectrogram_dataset``
    matches expected dataframe."""
    assert isinstance(df_returned, pd.DataFrame)

    assert df_returned.columns.values.tolist() == vak.prep.audio_dataset.DF_COLUMNS

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

    audio_paths_from_df = [Path(audio_path) for audio_path in df_returned.audio_path]

    # test that all audio files came from data_dir
    for audio_path_from_df in audio_paths_from_df:
        assert audio_path_from_df.parent == data_dir

    if expected_audio_paths:
        assert all(
            [
                expected_audio_path in audio_paths_from_df
                for expected_audio_path in expected_audio_paths
            ]
        )

    # if there are audio files we expect to **not** be in audio_path
    # -- because the associated annotations have labels not in labelset --
    # then test those files are **not** in spect_path
    if not_expected_audio_paths:
        assert all(
            [
                not_expected_audio_path not in audio_paths_from_df
                for not_expected_audio_path in not_expected_audio_paths
            ]
        )

    return True  # all asserts passed


@pytest.mark.parametrize(
    'data_dir, audio_format, labelset, annot_format, annot_file, expected_audio_paths, not_expected_audio_paths',
    [
        (AUDIO_DIR_CBIN, "cbin", LABELSET_NOTMAT, "notmat", None,
         AUDIO_LIST_CBIN_ALL_LABELS_IN_LABELSET, AUDIO_LIST_CBIN_LABELS_NOT_IN_LABELSET),
    ]
)
def test_prep_audio_dataset(
    data_dir, audio_format, labelset, annot_format, annot_file, tmp_path,
    expected_audio_paths, not_expected_audio_paths,
):
    """test that ``vak.prep.spectrogram_dataset.prep.prep_spectrogram_dataset`` works
    when we point it at directory of .cbin audio files
    and specify an annotation format"""
    dataset_df = vak.prep.audio_dataset.prep_audio_dataset(
        data_dir=data_dir,
        audio_format=audio_format,
        labelset=labelset,
        annot_format=annot_format,
        annot_file=annot_file,
    )

    assert returned_dataframe_matches_expected(
        dataset_df,
        data_dir=data_dir,
        labelset=labelset,
        annot_format=annot_format,
        audio_format=audio_format,
        annot_file=annot_file,
        expected_audio_paths=expected_audio_paths,
        not_expected_audio_paths=not_expected_audio_paths,
    )
