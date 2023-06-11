"""Helper functions for prep module.

There's two reasons for this module:
1. It lets us `import prep from prep` in vak/core/prep/__init__.py
so we don't have to write `vak.core.prep.prep.prep()`
2. It factors out smaller functions to unit test,
so that ``vak.core.prep.prep.prep` is less of a giant
imperative script.
"""
from __future__ import annotations

import logging
import pathlib
import shutil

import crowsetta
import numpy as np
import pandas as pd

from .. import (
    datasets,
    transforms
)


logger = logging.getLogger(__name__)


VALID_PURPOSES = frozenset(
    [
        "eval",
        "learncurve",
        "predict",
        "train",
    ]
)


def make_arrays_for_each_split(dataset_df: pd.DataFrame,
                               dataset_path: pathlib.Path,
                               purpose: str,
                               labelmap: dict | None = None,
                               annot_format: str | None = None,
                               ) -> None:
    """Makes arrays used by dataset classes,
    one array per split.

    We use arrays instead of loading files
    to minimize the amount of bookkeeping related to
    windows and the boundaries imposed by files;
    windowing becomes easier if we treat a set of files
    as a single time series since the only invalid indices
    for grabbing a window's worth of data are at the end
    of that time series, instead of at the end of
    every file. This does neglect edge effects
    that could be induced by where files start and stop
    in the single array. We assume these are rare
    in relation to the rest of the dataset.

    Parameters
    ----------
    dataset_df : pandas.Dataframe
    dataset_path : str, pathlib.Path
    purpose: str
    labelmap : dict
    annot_format : str
    """
    dataset_path = pathlib.Path(dataset_path)

    logger.info(f"Will use labelmap: {labelmap}")

    if annot_format:
        scribe = crowsetta.Transcriber(format=annot_format)

    for split in dataset_df.split:
        if split == 'None':
            # these are files that didn't get assigned to a split
            continue
        logger.info(f"Processing split: {split}")
        split_dst = dataset_path / split
        logger.info(f"Will save in: {split}")
        split_dst.mkdir(exist_ok=True)

        split_df = dataset_df[dataset_df.split == split]
        spect_paths = split_df['spect_path'].values
        # do this outside `if purpose != 'predict'` so we can iterate over
        # np.nans or Nones even if there's no annotations
        annot_paths = split_df['annot_path'].values

        logger.info(f"Loading data from {len(spect_paths)} spectrogram files "
              f"and {len(annot_paths)} annotation files")

        inputs, source_id_vec, inds_in_source_vec = [], [], []
        if purpose != 'predict':
            frame_labels, annots = [], []

        for source_id, (spect_path, annot_path) in enumerate(
                zip(spect_paths, annot_paths)
        ):
            # add to inputs
            d = np.load(spect_path)
            inputs.append(d["s"])

            # add to frame labels
            if purpose != 'predict':
                annot = scribe.from_file(annot_path).to_annot()
                annots.append(annot)  # we use this to save a .csv below
                lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
                lbls_frame = transforms.labeled_timebins.from_segments(
                    lbls_int,
                    annot.seq.onsets_s,
                    annot.seq.offsets_s,
                    d["t"],
                    unlabeled_label=labelmap["unlabeled"],
                )
                frame_labels.append(lbls_frame)

            # add to source_id and source_inds
            n_frames = d["t"].shape[-1]  # number of frames
            source_id_vec.append(
                np.ones((n_frames,)).astype(np.int32) * source_id
            )
            inds_in_source_vec.append(
                np.arange(n_frames)
            )

        inputs = np.concatenate(inputs, axis=1)
        source_id_vec = np.concatenate(source_id_vec)
        inds_in_source_vec = np.concatenate(inds_in_source_vec)
        if purpose != 'predict':
            frame_labels = np.concatenate(frame_labels)
            generic_seq = crowsetta.formats.seq.GenericSeq(annots=annots)

        logger.info(
            f"Saving ``inputs`` for frame classification dataset with size {round(inputs.nbytes * 1e-10, 2)} GB"
        )
        np.save(split_dst / datasets.frame_classification.INPUT_ARRAY_FILENAME, inputs)
        logger.info(
            "Saving ``source_id`` vector for frame classification dataset with size "
            f"{round(source_id_vec.nbytes * 1e-6, 2)} MB"
        )
        np.save(split_dst / datasets.frame_classification.SOURCE_IDS_ARRAY_FILENAME, source_id_vec)
        logger.info(
            "Saving ``inds_in_source_vec`` vector for frame classification dataset "
            f"with size {round(inds_in_source_vec.nbytes * 1e-6, 2)} MB"
        )
        np.save(split_dst / datasets.frame_classification.INDS_IN_SOURCE_ARRAY_FILENAME, inds_in_source_vec)
        if purpose != 'predict':
            logger.info(
                "Saving frame labels vector (targets) for frame classification dataset "
                f"with size {round(frame_labels.nbytes * 1e-6, 2)} MB"
            )
            np.save(split_dst / datasets.frame_classification.FRAME_LABELS_ARRAY_FILENAME, frame_labels)
            logger.info(
                "Saving annotations as csv"
            )
            generic_seq.to_file(split_dst / datasets.frame_classification.ANNOTATION_CSV_FILENAME)


def move_files_into_split_subdirs(dataset_df: pd.DataFrame, dataset_path: pathlib.Path, purpose: str) -> None:
    """Move files in dataset into sub-directories, one for each split in the dataset.

    This is run *after* calling :func:`vak.io.dataframe.from_files` to generate ``dataset_df``,
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

    # ---- first move all the spectrograms; we need to handle annotations separately -----------------------------------
    moved_spect_paths = []  # to clean up after moving -- may be empty if we copy all spects (e.g., user generated)
    # ---- copy/move files into split sub-directories inside dataset directory
    # Next line, note we drop any na rows in the split column, since they don't belong to a split anyway
    split_names = sorted(dataset_df.split.dropna().unique())

    for split_name in split_names:
        if split_name == 'None':
            # these are files that didn't get assigned to a split
            continue
        split_subdir = dataset_path / split_name
        split_subdir.mkdir()

        split_df = dataset_df[dataset_df.split == split_name].copy()
        split_spect_paths = [
            # this just converts from string to pathlib.Path
            pathlib.Path(spect_path)
            for spect_path in split_df['spect_path'].values
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

        # TODO: rewrite as 'moved_source_paths', etc., when we add audio
        new_spect_paths = []  # to fix DataFrame
        for spect_path in split_spect_paths:
            spect_path = pathlib.Path(spect_path)
            if move_spects:  # because it's within dataset_path already
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
                # rewrite paths relative to dataset directory's root, so dataset is portable
                pathlib.Path(new_spect_path).relative_to(dataset_path)
            )

        # cast to str before rewrite so that type doesn't silently change for some rows
        new_spect_paths = [str(new_spect_path) for new_spect_path in new_spect_paths]
        dataset_df.loc[split_df.index, 'spect_path'] = new_spect_paths

    if purpose != 'predict':
        if len(dataset_df["annot_path"].unique()) == 1:
            # --> there is a single annotation file associated with all rows
            # in this case we copy the single annotation file to the root of the dataset directory
            annot_path = pathlib.Path(
                dataset_df["annot_path"].unique().item()
            )
            copied_annot_path = dataset_path / annot_path.name
            # next code block:
            # if we converted some annotation format to normalize it,
            # e.g., birdsong-recognition-dataset,
            # and we already saved the converted annotations in dataset_path
            # we don't need to copy, and this would raise "it's the same file!" error. So we skip in that case
            if not copied_annot_path.exists():
                shutil.copy(src=annot_path, dst=copied_annot_path)
            # regardless of whether we copy, we want to write annot path relative to dataset directory root
            copied_annot_path = pathlib.Path(copied_annot_path).relative_to(dataset_path)
            dataset_df["annot_path"] = str(copied_annot_path)  # ensure string; one file -> same path for all rows

        elif len(dataset_df["annot_path"].unique()) == len(dataset_df):
            # --> there is a unique annotation file (path) for each row, i.e. a 1:1 mapping from spect:annotation
            # in this case we copy each annotation file to the split directory with its spectrogram file
            for split_name in sorted(dataset_df.split.unique()):
                split_subdir = dataset_path / split_name  # we already made this dir when moving spects
                split_df = dataset_df[dataset_df.split == split_name].copy()
                split_annot_paths = split_df['annot_path'].values.tolist()
                copied_annot_paths = []
                for annot_path in split_annot_paths:
                    annot_path = pathlib.Path(annot_path)
                    copied_annot_path = shutil.copy(
                        src=annot_path, dst=split_subdir
                    )
                    # rewrite paths relative to dataset directory's root, so dataset is portable
                    copied_annot_path = pathlib.Path(copied_annot_path).relative_to(dataset_path)
                    copied_annot_paths.append(copied_annot_path)

                # cast back to str before rewrite so that type doesn't silently change for some rows
                copied_annot_paths = [
                    str(copied_annot_path) for copied_annot_path in copied_annot_paths
                ]
                dataset_df.loc[split_df.index, 'annot_path'] = copied_annot_paths

        else:
            raise ValueError(
                "Unable to load labels from dataframe; did not find an annotation file for each row or "
                "a single annotation file associated with all rows."
            )

    # ---- clean up after moving/copying -------------------------------------------------------------------------------
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
