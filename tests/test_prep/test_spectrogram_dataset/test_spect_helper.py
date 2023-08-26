"""tests for ``vak.prep.spectrogram_dataset.spect_helper`` module"""
from pathlib import Path

import pandas as pd
import pytest

import vak.prep.spectrogram_dataset.spect_helper
import vak.common.files.spect


def spect_paths_from_df_as_paths(dataset_df):
    return [Path(spect_path) for spect_path in dataset_df["spect_path"]]


def assert_expected_spect_paths_in_dataframe(
    spect_paths_from_df, spect_format, expected_spect_paths, not_expected_spect_paths=None
):
    """Tests that a dataframe ``dataset_df`` contains one file
    for each path in ``expected_spect_paths`` in its ``spect_path`` column,
    and only those paths.

    Parameters
    ----------
    dataset_df : pandas.Dataframe
        created by vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files
    expected_spect_paths : list
        of paths to spectrogram files, that **should** be in dataset_df.spect_path column
    not_expected_spect_paths : list
        of paths to spectrogram files, that should **not** be in dataset_df.spect_path column
    """
    spect_file_names_from_df = [spect_path.name for spect_path in spect_paths_from_df]

    if spect_format == 'mat':
        expected_spectfile_names = [
            spect_path.name.replace('.mat', '.npz')
            for spect_path in expected_spect_paths
        ]
    else:
        expected_spectfile_names = [
            spect_path.name for spect_path in expected_spect_paths
        ]

    assert all(
        [expected_spect_file in spect_file_names_from_df for expected_spect_file in expected_spectfile_names]
    )

    # test that **only** expected paths were in DataFrame
    if not_expected_spect_paths is not None:
        if spect_format == 'mat':
            not_expected_spectfile_names = [
                spect_path.name.replace('.mat', '.npz')
                for spect_path in not_expected_spect_paths
            ]
        else:
            not_expected_spectfile_names = [
                spect_path.name for spect_path in not_expected_spect_paths
            ]
        assert all(
            [not_expected_spect_file not in spect_file_names_from_df
             for not_expected_spect_file in not_expected_spectfile_names]
        )


@pytest.mark.parametrize(
    "spect_format, annot_format, spect_ext, labelset, arg_to_test",
    [
        ("mat", "yarden", None, True, 'spect_dir'),
        ("npz", "notmat", ".spect.npz", True, 'spect_dir'),
        ("mat", "yarden", None, False, 'spect_dir'),
        ("npz", "notmat", ".spect.npz", False, 'spect_dir'),
        ("mat", None, None, False, 'spect_dir'),
        ("npz", None, ".spect.npz", False, 'spect_dir'),

        ("mat", "yarden", None, True, 'spect_files'),
        ("npz", "notmat", ".spect.npz", True, 'spect_files'),
        ("mat", "yarden", None, False, 'spect_files'),
        ("npz", "notmat", ".spect.npz", False, 'spect_files'),
        ("mat", None, None, False, 'spect_files'),
        ("npz", None, ".spect.npz", False, 'spect_files'),
    ],
)
def test_make_dataframe_of_spect_files(
    spect_format,
    annot_format,
    spect_ext,
    labelset,
    arg_to_test,
    specific_spect_dir,
    specific_spect_list,
    specific_annot_list,
    specific_labelset,
    tmp_path,
):
    """Test that ``vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files`` works
    when we point it at directory + give it list of annotations"""
    if arg_to_test == 'spect_dir':
        spect_dir = specific_spect_dir(spect_format)
        spect_files = None
    elif arg_to_test == 'spect_files':
        spect_dir = None
        spect_files = specific_spect_list(spect_format)

    if labelset:
        labelset = specific_labelset(annot_format)
    else:
        labelset = None

    if annot_format:
        annot_list = specific_annot_list(annot_format)
    else:
        annot_list = None

    if spect_format == "mat":
        spect_output_dir = tmp_path
    else:
        spect_output_dir = None

    dataset_df = vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
        spect_format=spect_format,
        spect_dir=spect_dir,
        spect_files=spect_files,
        spect_output_dir=spect_output_dir,
        labelset=labelset,
        annot_list=annot_list,
        annot_format=annot_format,
        spect_ext=spect_ext,
    )
    assert type(dataset_df) == pd.DataFrame

    spect_paths_from_df = spect_paths_from_df_as_paths(dataset_df)
    if labelset:
        expected_spect_list = specific_spect_list(
            spect_format, "all_labels_in_labelset"
        )
        not_expected_spect_list = specific_spect_list(
            spect_format, "labels_not_in_labelset"
        )
    else:
        expected_spect_list = specific_spect_list(spect_format)
        not_expected_spect_list = None

    assert_expected_spect_paths_in_dataframe(
        spect_paths_from_df, spect_format,
        expected_spect_list, not_expected_spect_list
    )

    if spect_format == 'mat':
        expected_parent = spect_output_dir
    else:
        expected_parent = specific_spect_dir(spect_format)
    assert all(
        [spect_path.parent == expected_parent for spect_path in spect_paths_from_df]
    )


def test_make_dataframe_of_spect_files_no_spect_dir_files_or_map_raises(annot_list_yarden):
    """test that calling ``make_dataframe_of_spect_files`` without one of:
    spect dir, spect files, or spect files/annotations mapping
    raises ValueError"""
    with pytest.raises(ValueError):
        vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
            spect_format="mat",
            spect_dir=None,
            spect_files=None,
            annot_list=annot_list_yarden,
            annot_format="yarden",
        )


def test_make_dataframe_of_spect_files_invalid_spect_format_raises(spect_dir_mat, annot_list_yarden):
    """test that calling ``make_dataframe_of_spect_files`` with an invalid spect format raises a ValueError"""
    with pytest.raises(ValueError):
        vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
            spect_format="npy",  # 'npy' not a valid spect format
            spect_dir=spect_dir_mat,
            annot_list=annot_list_yarden,
            annot_format="yarden",
        )


def test_make_dataframe_of_spect_files_dir_and_list_raises(
    spect_dir_mat, spect_list_mat, annot_list_yarden
):
    """test that calling ``make_dataframe_of_spect_files`` with both dir and list raises a ValueError"""
    with pytest.raises(ValueError):
        vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
            spect_format="mat",
            spect_dir=spect_dir_mat,
            spect_files=spect_list_mat,
            annot_list=annot_list_yarden,
            annot_format="yarden",
        )


def test_make_dataframe_of_spect_files_annot_list_without_annot_format_raises(
    spect_dir_mat, spect_list_mat, annot_list_yarden
):
    """test that calling ``make_dataframe_of_spect_files`` with a list of annotations
    but no annot_format raises a ValueError"""
    with pytest.raises(ValueError):
        vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
            spect_format="mat", annot_list=annot_list_yarden, annot_format=None
        )
