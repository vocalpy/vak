"""Helper functions for prep module.

There's two reasons for this module:
1. It lets us `import prep from prep` in vak/core/prep/__init__.py
so we don't have to write `vak.core.prep.prep.prep()`
2. It factors out smaller functions to unit test,
so that ``vak.core.prep.prep.prep` is less of a giant
imperative script.
"""
from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pandas as pd


VALID_PURPOSES = frozenset(
    [
        "eval",
        "learncurve",
        "predict",
        "train",
    ]
)


def move_files_into_split_subdirs(dataset_df: pd.DataFrame, dataset_path: pathlib.Path, purpose: str) -> None:
    """Move files in dataset into sub-directories, one for each split in the dataset.

    This is run *after* calling `vak.io.dataframe` to generate ``dataset_df``,
    to avoid coupling the generation of the dataframe to organization of the dataset.

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        A ``pandas.DataFrame`` returned by :func:`vak.io.dataframe.from_files`
        with a ``'split'`` column added, as a result of calling
        :func:`vak.io.dataframe.from_files` or because it was added "manually"
        by calling :func:`vak.core.prep.prep_helper.add_split_col` (as is done
        for 'predict' when the entire ``DataFrame`` belongs to this
        "split").
    dataset_path : pathlib.Path
        Path to directory that represents dataset.
    purpose: str
        A string indicating what the dataset will be used for.
        One of {'train', 'eval', 'predict', 'learncurve'}.
        Determined by :func:`vak.core.prep.prep`
        using the TOML configuration file.

    Returns
    -------
    None

    The ``DataFrame`` is modified in place
    as the files are moved, so nothing is returned.
    """
    moved_spect_paths = []  # to clean up after moving -- may be empty if we copy all spects (e.g., user generated)
    # ---- copy/move files into split sub-directories inside dataset directory
    for split_name in sorted(dataset_df.split.unique()):
        split_subdir = dataset_path / split_name
        split_subdir.mkdir()

        split_df = dataset_df[dataset_df.split == split_name].copy()
        split_spect_paths = [
            pathlib.Path(spect_path)
            for spect_path in split_df['spect_path'].values
        ]
        to_move = [
            # if dataset_path is one of the parents of spect_path, we can move; otherwise, we copy
            dataset_path.resolve() in list(spect_path.parents)
            for spect_path in split_spect_paths
        ]

        # TODO: rewrite as 'moved_source_paths', etc., when we add audio
        new_spect_paths = []  # to fix DataFrame
        for spect_path, move_spect in zip(split_spect_paths, to_move):
            spect_path = pathlib.Path(spect_path)
            if move_spect:  # because it's within dataset_path already
                new_spect_path = spect_path.rename(
                    split_subdir / spect_path.name
                )
                moved_spect_paths.append(
                    spect_path
                )
            else:  # copy instead of moving
                new_spect_path = shutil.copy(
                    src=spect_path, dst=split_subdir
                )

            new_spect_paths.append(
                new_spect_path
            )

        # cast to str before rewrite so that type doesn't silently change for some rows
        new_spect_paths = [str(new_spect_path) for new_spect_path in new_spect_paths]
        dataset_df.loc[split_df.index, 'spect_path'] = new_spect_paths

        if purpose != 'predict':
            split_annot_paths = split_df['annot_path'].values.tolist()
            copied_annot_paths = []
            for annot_path in split_annot_paths:
                annot_path = pathlib.Path(annot_path)
                copied_annot_paths.append(
                    shutil.copy(
                        src=annot_path, dst=split_subdir
                    )
                )
            # cast to str before rewrite so that type doesn't silently change for some rows
            copied_annot_paths = [str(copied_annot_path) for copied_annot_path in copied_annot_paths]
            dataset_df.loc[split_df.index, 'annot_path'] = copied_annot_paths

    # ---- clean up after moving/copying
    # remove any directories that we just emptied
    if moved_spect_paths:
        unique_parents = set([
            moved_spect.parent for moved_spect in moved_spect_paths
        ])
        for parent in unique_parents:
            if len(list(parent.iterdir())) < 1:
                shutil.rmtree(parent)


def get_dataset_csv_filename(data_dir_name: str, timenow: str) -> str:
    """Get name of csv file representing dataset.

    This function is called by
    :func:`vak.core.prep.get_dataset_csv_path`.

    Parameters
    ----------
    data_dir_name : str
        Name of directory specified as parameter ``data_dir``
        when calling :func:`vak.core.prep.prep`.
        This becomes the "prefix" of the csv filename.
    timenow : str
        Timestamp.
        This becomes the "suffix" of the csv filename.

    Returns
    -------
    dataset_csv_filename : str
        String, in the form f"{data_dir_name}_prep_{timenow}.csv"
    """
    return f"{data_dir_name}_prep_{timenow}.csv"


def get_dataset_csv_path(dataset_path: pathlib.Path, data_dir_name: str, timenow: str) -> pathlib.Path:
    """Returns the path that :func:`vak.core.prep.prep` should use to save
    a pandas DataFrame representing a dataset to a csv file.

    This function is called by :func:`vak.core.prep.prep`.

    Parameters
    ----------
    dataset_path : str, pathlib.Path
        Path to directory that represents dataset.
    data_dir_name : str
        Name of directory specified as parameter ``data_dir``
        when calling :func:`vak.core.prep.prep`.
        This becomes the "prefix" of the csv filename.
    timenow : str
        Timestamp.
        This becomes the "suffix" of the csv filename.

    Returns
    -------
    dataset_csv_path : pathlib.Path
        Path that is used when saving ``dataset_df`` as a csv file
        in the root of the dataset directory, ``dataset_path``.
    """
    dataset_csv_filename = get_dataset_csv_filename(data_dir_name, timenow)
    dataset_csv_path = dataset_path / dataset_csv_filename
    return dataset_csv_path


def add_split_col(df: pd.DataFrame, split: str) -> None:
    """Add a 'split' column to a pandas DataFrame.

    Used by :func:`vak.prep`
    to assign an entire dataset to the same split,
    e.g. 'train' or 'predict'.
    All rows in the 'split' column will have the value specified.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe that represents a dataset.
    split : str
        A string that will be assigned to every row
        in the added "split" column.
        One of {'train', 'val', 'test', 'predict'}.
    """
    if split not in {"train", "val", "test", "predict"}:
        raise ValueError(
            f"value for split should be one of {{'train', 'val', 'test', 'predict'}}, but was '{split}'"
        )
    split_col = np.asarray([split for _ in range(len(df))], dtype="object")
    df["split"] = split_col
    return df


def validate_and_get_timebin_dur(df: pd.DataFrame, expected_timebin_dur: float | None = None) -> float:
    """Validate that there is a single, unique value for the time bin duration of all
    spectrograms in a dataset. If so, return that value.

    The dataset is represented as a pandas DataFrame.

    Parameters
    ----------
    df : pandas.Dataframe
        A pandas.DataFrame created by
        ``vak.io.dataframe.from_files``
        or ``vak.io.spect.to_dataframe``.
    expected_timebin_dur : float

    Returns
    -------
    timebin_dur : float
        The duration of a time bin in seconds
        for all spectrograms in the dataset.
    """
    timebin_dur = df["timebin_dur"].unique()
    if len(timebin_dur) > 1:
        raise ValueError(
            f"Found more than one time bin duration in dataset: {timebin_dur}"
        )
    elif len(timebin_dur) == 1:
        timebin_dur = timebin_dur.item()

    if expected_timebin_dur:
        if timebin_dur != expected_timebin_dur:
            raise ValueError(
                f"Timebin duration from dataset, {timebin_dur}, "
                f"did not match expected timebin duration, {expected_timebin_dur}."
            )

    return timebin_dur
