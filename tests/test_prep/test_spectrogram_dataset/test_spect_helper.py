"""tests for ``vak.prep.spectrogram_dataset.spect_helper`` module"""
from pathlib import Path

import pandas as pd
import pytest

import vak.prep.spectrogram_dataset.spect_helper
import vak.common.files.spect


def expected_spect_paths_in_dataframe(
    dataset_df, expected_spect_paths, not_expected_spect_paths=None
):
    """tests that a dataframe ``dataset_df`` contains
    all paths in ``expected_spect_paths``, and only those paths,
    in its ``spect_path`` column.
    If so, returns True.

    Parameters
    ----------
    dataset_df : pandas.Dataframe
        created by vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files
    expected_spect_paths : list
        of paths to spectrogram files, that **should** be in dataset_df.spect_path column
    not_expected_spect_paths : list
        of paths to spectrogram files, that should **not** be in dataset_df.spect_path column
    """
    assert type(dataset_df) == pd.DataFrame

    spect_paths_from_df = [Path(spect_path) for spect_path in dataset_df["spect_path"]]

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


@pytest.mark.parametrize(
    "spect_format, annot_format, spect_ext",
    [
        ("mat", "yarden", None),
        ("npz", "notmat", ".spect.npz"),
    ],
)
def test_make_dataframe_of_spect_files_spect_dir(
    spect_format,
    annot_format,
    spect_ext,
    specific_spect_dir,
    specific_spect_list,
    specific_annot_list,
    specific_labelset,
):
    """test that ``vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files`` works
    when we point it at directory + give it list of annotations"""
    spect_dir = specific_spect_dir(spect_format)
    labelset = specific_labelset(annot_format)
    annot_list = specific_annot_list(annot_format)

    dataset_df = vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
        spect_format=spect_format,
        spect_dir=spect_dir,
        labelset=labelset,
        annot_list=annot_list,
        annot_format=annot_format,
        spect_ext=spect_ext,
    )

    spect_list_all_labels_in_labelset = specific_spect_list(
        spect_format, "all_labels_in_labelset"
    )
    spect_list_labels_not_in_labelset = specific_spect_list(
        spect_format, "labels_not_in_labelset"
    )
    assert expected_spect_paths_in_dataframe(
        dataset_df, spect_list_all_labels_in_labelset, spect_list_labels_not_in_labelset
    )


@pytest.mark.parametrize(
    "spect_format, annot_format, spect_ext",
    [
        ("mat", "yarden", None),
        ("npz", "notmat", ".spect.npz"),
    ],
)
def test_make_dataframe_of_spect_files_spect_dir_no_labelset(
    spect_format,
    annot_format,
    spect_ext,
    specific_spect_dir,
    specific_spect_list,
    specific_annot_list,
):
    """test that ``vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files`` works when we point it at directory + give it list of annotations
    but do not give it a labelset to filter out files"""
    spect_dir = specific_spect_dir(spect_format)
    annot_list = specific_annot_list(annot_format)

    dataset_df = vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
        spect_format=spect_format,
        spect_dir=spect_dir,
        labelset=None,
        annot_list=annot_list,
        annot_format="yarden",
        spect_ext=spect_ext,
    )

    spect_list = specific_spect_list(spect_format)
    assert expected_spect_paths_in_dataframe(dataset_df, spect_list)


@pytest.mark.parametrize(
    "spect_format, annot_format, spect_ext",
    [
        ("mat", "yarden", None),
        ("npz", "notmat", ".spect.npz"),
    ],
)
def test_make_dataframe_of_spect_files_spect_dir_without_annot(
    spect_format, annot_format, spect_ext, specific_spect_dir, specific_spect_list
):
    """test ``vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files`` works with a dataset from spectrogram files without annotations,
    # e.g. if we're going to predict the annotations using the spectrograms"""
    spect_dir = specific_spect_dir(spect_format)

    dataset_df = vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
        spect_format=spect_format, spect_dir=spect_dir, annot_list=None, spect_ext=spect_ext,
    )

    spect_list = specific_spect_list(spect_format)
    assert expected_spect_paths_in_dataframe(dataset_df, spect_list)


@pytest.mark.parametrize(
    "spect_format, annot_format, spect_ext",
    [
        ("mat", "yarden", None),
        ("npz", "notmat", ".spect.npz"),
    ],
)
def test_make_dataframe_of_spect_files_spect_files(
    spect_format,
    annot_format,
    spect_ext,
    specific_spect_list,
    specific_annot_list,
    specific_labelset,
):
    """test that ``vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files`` works
    when we give it list of spectrogram files and a list of annotations"""
    spect_list = specific_spect_list(spect_format)
    labelset = specific_labelset(annot_format)
    annot_list = specific_annot_list(annot_format)

    dataset_df = vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
        spect_format=spect_format,
        spect_files=spect_list,
        labelset=labelset,
        annot_list=annot_list,
        annot_format=annot_format,
        spect_ext=spect_ext,
    )

    spect_list_all_labels_in_labelset = specific_spect_list(
        spect_format, "all_labels_in_labelset"
    )
    spect_list_labels_not_in_labelset = specific_spect_list(
        spect_format, "labels_not_in_labelset"
    )
    assert expected_spect_paths_in_dataframe(
        dataset_df, spect_list_all_labels_in_labelset, spect_list_labels_not_in_labelset
    )


@pytest.mark.parametrize(
    "spect_format, annot_format, spect_ext",
    [
        ("mat", "yarden", None),
        ("npz", "notmat", ".spect.npz"),
    ],
)
def test_make_dataframe_of_spect_files_spect_files_no_labelset(
    spect_format, annot_format, spect_ext, specific_spect_list, specific_annot_list
):
    """test that ``vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files`` works
    when we give it list of spectrogram files and a list of annotations
    but do not give it a labelset to filter out files"""
    spect_list = specific_spect_list(spect_format)
    annot_list = specific_annot_list(annot_format)

    dataset_df = vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
        spect_format=spect_format,
        spect_files=spect_list,
        labelset=None,
        annot_list=annot_list,
        annot_format=annot_format,
        spect_ext=spect_ext,
    )

    spect_list = specific_spect_list(spect_format)
    assert expected_spect_paths_in_dataframe(dataset_df, spect_list)


@pytest.mark.parametrize(
    "spect_format, annot_format, spect_ext",
    [
        ("mat", "yarden", None),
        ("npz", "notmat", ".spect.npz"),
    ],
)
def test_make_dataframe_of_spect_files_spect_annot_map(
    spect_format,
    annot_format,
    spect_ext,
    specific_spect_list,
    specific_annot_list,
    specific_labelset,
):
    """test that ``vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files`` works
    when we give it a dict that maps spectrogram files to annotations
    but do not give it a labelset to filter out files"""
    spect_list = specific_spect_list(spect_format)
    labelset = specific_labelset(annot_format)
    annot_list = specific_annot_list(annot_format)

    spect_annot_map = dict(zip(spect_list, annot_list))
    dataset_df = vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
        spect_format=spect_format,
        labelset=labelset,
        spect_annot_map=spect_annot_map,
        annot_format=annot_format,
        spect_ext=spect_ext,
    )

    spect_list_all_labels_in_labelset = specific_spect_list(
        spect_format, "all_labels_in_labelset"
    )
    spect_list_labels_not_in_labelset = specific_spect_list(
        spect_format, "labels_not_in_labelset"
    )
    assert expected_spect_paths_in_dataframe(
        dataset_df, spect_list_all_labels_in_labelset, spect_list_labels_not_in_labelset
    )


@pytest.mark.parametrize(
    "spect_format, annot_format, spect_ext",
    [
        ("mat", "yarden", None),
        ("npz", "notmat", ".spect.npz"),
    ],
)
def test_make_dataframe_of_spect_files_spect_annot_map_no_labelset(
    spect_format, annot_format, spect_ext, specific_spect_list, specific_annot_list
):
    """test that ``vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files`` works
    when we give it a dict that maps spectrogram files to annotations
    but do not give it a labelset to filter out files"""
    spect_list = specific_spect_list(spect_format)
    annot_list = specific_annot_list(annot_format)

    spect_annot_map = dict(zip(spect_list, annot_list))

    dataset_df = vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
        spect_format=spect_format,
        labelset=None,
        spect_annot_map=spect_annot_map,
        annot_format=annot_format,
        spect_ext=spect_ext,
    )

    spect_list = specific_spect_list(spect_format)
    assert expected_spect_paths_in_dataframe(dataset_df, spect_list)


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
            spect_annot_map=None,
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


def test_make_dataframe_of_spect_files_dir_and_map_raises(
    spect_dir_mat, spect_list_mat, annot_list_yarden
):
    """test that calling ``make_dataframe_of_spect_files`` with both dir and map raises a ValueError"""
    spect_annot_map = dict(zip(spect_list_mat, annot_list_yarden))
    with pytest.raises(ValueError):
        vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
            spect_format="mat",
            spect_dir=spect_dir_mat,
            spect_annot_map=spect_annot_map,
            annot_format="yarden",
        )


def test_make_dataframe_of_spect_files_list_and_map_raises(
    spect_dir_mat, spect_list_mat, annot_list_yarden
):
    """test that calling ``make_dataframe_of_spect_files`` with both list and map raises a ValueError"""
    spect_annot_map = dict(zip(spect_list_mat, annot_list_yarden))
    with pytest.raises(ValueError):
        vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
            spect_format="mat",
            spect_files=spect_list_mat,
            spect_annot_map=spect_annot_map,
            annot_format="yarden",
        )


def test_make_dataframe_of_spect_files_annot_list_and_map_raises(
    spect_dir_mat, spect_list_mat, annot_list_yarden
):
    """test that calling ``make_dataframe_of_spect_files`` with both list of annotations and map raises a ValueError"""
    spect_annot_map = dict(zip(spect_list_mat, annot_list_yarden))
    with pytest.raises(ValueError):
        vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
            spect_format="mat",
            spect_annot_map=spect_annot_map,
            annot_list=annot_list_yarden,
            annot_format="yarden",
        )


def test_make_dataframe_of_spect_files_annot_list_without_annot_format_raises(
    spect_dir_mat, spect_list_mat, annot_list_yarden
):
    """test that calling ``make_dataframe_of_spect_files`` with a list of annotations but no annot_format raises a ValueError"""
    spect_annot_map = dict(zip(spect_list_mat, annot_list_yarden))
    with pytest.raises(ValueError):
        vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
            spect_format="mat", annot_list=annot_list_yarden, annot_format=None
        )


def test_make_dataframe_of_spect_files_spect_annot_map_without_annot_format_raises(
    spect_dir_mat, spect_list_mat, annot_list_yarden
):
    """test that calling ``make_dataframe_of_spect_files`` with a list of annotations but no annot_format raises a ValueError"""
    spect_annot_map = dict(zip(spect_list_mat, annot_list_yarden))
    with pytest.raises(ValueError):
        vak.prep.spectrogram_dataset.spect_helper.make_dataframe_of_spect_files(
            spect_format="mat", spect_annot_map=spect_annot_map, annot_format=None
        )
