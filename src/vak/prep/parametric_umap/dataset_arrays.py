"""Helper functions for `vak.prep.dimensionality_reduction` module
that handle array files.
"""

from __future__ import annotations

import logging
import pathlib
import shutil

import pandas as pd

logger = logging.getLogger(__name__)


def move_files_into_split_subdirs(
    dataset_df: pd.DataFrame, dataset_path: pathlib.Path, purpose: str
) -> None:
    """Move npy files in dataset into sub-directories, one for each split in the dataset.

    This is run *after* calling :func:`vak.prep.unit_dataset.prep_unit_dataset`
    to generate ``dataset_df``.

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        A ``pandas.DataFrame`` returned by
        :func:`vak.prep.unit_dataset.prep_unit_dataset`
        with a ``'split'`` column added, as a result of calling
        :func:`vak.prep.split.unit_dataframe` or because it was added "manually"
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
    moved_spect_paths = (
        []
    )  # to clean up after moving -- may be empty if we copy all spects (e.g., user generated)
    # ---- copy/move files into split sub-directories inside dataset directory
    # Next line, note we drop any na rows in the split column, since they don't belong to a split anyway
    split_names = sorted(dataset_df.split.dropna().unique())

    for split_name in split_names:
        if split_name == "None":
            # these are files that didn't get assigned to a split
            continue
        split_subdir = dataset_path / split_name
        split_subdir.mkdir()

        split_df = dataset_df[dataset_df.split == split_name].copy()
        split_spect_paths = [
            # this just converts from string to pathlib.Path
            pathlib.Path(spect_path)
            for spect_path in split_df["spect_path"].values
        ]
        is_in_dataset_dir = [
            # if dataset_path is one of the parents of spect_path, we can move; otherwise, we copy
            dataset_path.resolve() in list(spect_path.parents)
            for spect_path in split_spect_paths
        ]
        if all(is_in_dataset_dir):
            move_spects = True
        elif all([not is_in_dir for is_in_dir in is_in_dataset_dir]):
            move_spects = False
        else:
            raise ValueError(
                "Expected to find either all spectrograms were in dataset directory, "
                "or all were in some other directory, but found a mixture. "
                f"Spectrogram paths for split being moved within dataset directory:\n{split_spect_paths}"
            )

        new_spect_paths = []  # to fix DataFrame
        for spect_path in split_spect_paths:
            spect_path = pathlib.Path(spect_path)
            if move_spects:  # because it's within dataset_path already
                new_spect_path = spect_path.rename(
                    split_subdir / spect_path.name
                )
                moved_spect_paths.append(spect_path)
            else:  # copy instead of moving
                new_spect_path = shutil.copy(src=spect_path, dst=split_subdir)

            new_spect_paths.append(
                # rewrite paths relative to dataset directory's root, so dataset is portable
                pathlib.Path(new_spect_path).relative_to(dataset_path)
            )

        # cast to str before rewrite so that type doesn't silently change for some rows
        new_spect_paths = [
            str(new_spect_path) for new_spect_path in new_spect_paths
        ]
        dataset_df.loc[split_df.index, "spect_path"] = new_spect_paths

    # ---- clean up after moving/copying -------------------------------------------------------------------------------
    # remove any directories that we just emptied
    if moved_spect_paths:
        unique_parents = set(
            [moved_spect.parent for moved_spect in moved_spect_paths]
        )
        for parent in unique_parents:
            if len(list(parent.iterdir())) < 1:
                shutil.rmtree(parent)
