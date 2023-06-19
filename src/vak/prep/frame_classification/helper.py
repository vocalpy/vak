"""Helper functions for frame classification dataset prep."""
from __future__ import annotations

import collections
import copy
import logging
import pathlib
import shutil

import crowsetta
import numpy as np
import numpy.typing as npt
import pandas as pd

from ... import (
    common,
    datasets,
    transforms
)


logger = logging.getLogger(__name__)


def sort_source_paths_and_annots_by_label_freq(
        source_paths: list[str] | npt.NDArray,
        annots: list[crowsetta.Annotation]
) -> tuple[list[str], list[crowsetta.Annotation]]:
    """Sort source paths and annotations by frequency of labels
    in annotations.

   :func:`vak.prep.frame_classification.helper.make_frame_classification_arrays_from_spect_and_annot_paths`
   uses this function to sort before cropping a dataset to a specified duration,
   so that it's less likely that cropping will remove all occurrences of any label class
   from the total dataset.

    Parameters
    ----------
    source_paths : list, np.ndarray
    annots: list

    Returns
    -------
    source_paths_sorted : list
    annots_sorted: list
    """
    if isinstance(source_paths, np.ndarray):
        source_paths = source_paths.tolist()

    if not(
        len(source_paths) == len(annots)
    ):
        raise ValueError(
            f"``source_paths`` and ``annots`` have different lengths:"
            f"len(source_paths)={len(source_paths)},"
            f"len(annots)={len(annots)}"
        )

    all_labels = [
        lbl for annot in annots for lbl in annot.seq.labels
    ]
    label_counts = collections.Counter(all_labels)

    # we copy so we can remove items to make sure we append all below
    source_paths_copy = copy.deepcopy(source_paths)
    annots_copy = copy.deepcopy(annots)
    source_paths_sorted, annots_sorted = [], []
    for label, _ in reversed(label_counts.most_common()):
        for source_path, annot in zip(source_paths_copy, annots_copy):
            if label in annot.seq.labels.tolist():
                annots_sorted.append(annot)
                source_paths_sorted.append(source_path)
                annots_copy.remove(annot)
                source_paths_copy.remove(source_path)

    # make sure we got all source_paths + annots
    if len(annots_copy) > 0:
        for source_path, annot in zip(source_paths_copy, annots_copy):
            annots_sorted.append(annot)
            source_paths_sorted.append(source_path)
            annots_copy.remove(annot)
            source_paths_copy.remove(source_path)

    if len(annots_copy) > 0:
        raise ValueError(
            "Not all ``annots`` were used in sorting."
            f"Leftover ``annots``: {annots_copy}"
        )

    if not (
            len(annots_sorted) == len(source_paths_sorted) == len(annots)
    ):
        raise ValueError(
            "Inconsistent lengths after sorting:"
            "len(annots_sorted) == len(source_paths_sorted) == len(annots)"
        )

    return source_paths_sorted, annots_sorted


def crop_arrays_keep_classes(
        inputs: npt.NDArray,
        source_ids: npt.NDArray,
        crop_dur: float,
        frame_dur: float,
        labelmap: dict | None = None,
        frame_labels: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    r"""Crop vectors representing WindowDataset
    to a target duration.

    This function "crops" a WindowDataset
    by shortening the vectors that represent
    valid windows in a way that
    ensures all classes are still present after cropping.
    It first tries to crop from the end of the dataset,
    then from the front,
    and then finally it tries to remove
    unlabeled periods that are at least equal to
    ``window_size`` + 2 time bins, until
    the total duration reaches the target size.
    If none of those approaches can preserve all classes
    in the dataset, the function raises an error.

    Parameters
    ----------
    inputs : numpy.ndarray
        Inputs to neural network that performs
        frame classification task.
        Must be 1-D or 2-D, either an array of audio
        data or spectrograms.
    source_ids : numpy.ndarray
        Represents the "ID" of any source file,
        i.e., the index into ``spect_paths``
        that will let us load that file.
        For a dataset with :math:`m` files,
        this will be an array of length :math:`T`,
        the total number of time bins across all files,
        with elements :math:`i in (0, 1, ..., m - 1)`
        indicating which time bins
        correspond to which file :math:`m_i`:
         :math:`(0, 0, 0, ..., 1, 1, ..., m - 1, m -1)`.
    crop_dur : float
        Duration to which dataset should be "cropped", in seconds.
    timebin_dur : float
        For a dataset of audio,
        the duration of a single sample,
        i.e., the inverse of the sampling rate given in classesHertz.
        For a dataset of spectrograms,
        the duration of a single time bin in the spectrograms.
    frame_labels : numpy.ndarray, optional
        Vector of labels for frames,
        where labels are from
        the set of values in ``labelmap``.
        Optional, default is None.
        If ``frame_labels`` is specified
        then ``labelmap`` must also be specified,
        and a ValueError will be raised
        if all classes in ``labelmap``
        are not present after cropping.
    labelmap : dict, optional
        Dict that maps labels from dataset
        to a series of consecutive integers.
        To create a label map, pass a set of labels
        to the :func:`vak.common.labels.to_map` function.
        Optional, default is None.
        If ``frame_labels`` is specified
        then ``labelmap`` must also be specified,
        and a ValueError will be raised
        if all classes in ``labelmap``
        are not present after cropping.

    Returns
    -------
    inputs_cropped : numpy.ndarray
        The ``inputs`` array after cropping.
    sourc_id_cropped : numpy.ndarray
        The ``source_ids`` vector after cropping.
    frame_labels_cropped : numpy.ndarray
        The ``frame_labels`` vector after cropping,
        if ``frame_labels`` was provided as an argument.
    """
    # ---- pre-conditions
    # check that all are numpy arrays
    # and check that inputs is 1d or 2d
    if not inputs.ndim in (1, 2):
        raise ValueError(
            f"The ``inputs`` array must be 1- or 2- dimensional but ``inputs.ndim`` was: {inputs.ndim}"
        )
    source_ids = common.validators.column_or_1d(source_ids)
    if frame_labels is not None:
        frame_labels = common.validators.column_or_1d(frame_labels)

    lens = [
        inputs.shape[-1],
        source_ids.shape[-1]
    ]
    if frame_labels is not None:
        lens.append(frame_labels.shape[-1])
    uniq_lens = set(lens)
    if len(uniq_lens) != 1:
        raise ValueError(
            "``inputs``, ``source_ids``, and ``frame_labels`` (if provided) should all "
            "have the same length, but ``crop_to_dur`` did not find one unique length. "
            "Lengths of ``inputs``, ``source_ids``, and ``frame_labels`` (if provided): "
            f"were: {lens}"
        )
    else:
        length = uniq_lens.pop()

    # ---- compute target length in number of time bins
    cropped_length = np.round(crop_dur / frame_dur).astype(int)

    if length == cropped_length:
        # already correct length
        return inputs, source_ids, frame_labels

    elif length < cropped_length:
        raise ValueError(
            f"Arrays have length {length} "
            f"that is shorter than correct length, {cropped_length}, "
            f"(= target duration {crop_dur} / duration of a single frame, {frame_dur})."
        )

    # ---- Do the cropping ----------------------------------------
    class_labels = set(labelmap.values())
    if frame_labels is not None:
        frame_labels_cropped = frame_labels[:cropped_length]
        uniq_frame_labels_cropped = set(np.unique(frame_labels_cropped))

        if uniq_frame_labels_cropped == class_labels:
            return inputs[:cropped_length], source_ids[:cropped_length], frame_labels
        else:
            raise ValueError(
                f"Unable to crop dataset of duration {length * frame_dur} to ``crop_dur`` {crop_dur} "
                f"and maintain occurrence of all labels in ``labelmap``."
            )
    else:
        return inputs[:cropped_length], source_ids[:cropped_length], frame_labels  # frame_labels is None here


def make_frame_classification_arrays_from_source_paths_and_annots(
        source_paths: list[str],
        labelmap: dict | None = None,
        annots: list[crowsetta.Annotation] | None = None,
        crop_dur: float | None = None,
        frame_dur: float | None = None,
        spect_key: str = "s",
        timebins_key: str = "t",
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]:
    """Makes arrays used by dataset classes
    for frame classification task
    from a list of audio or spectrogram paths
    and an optional, paired list of annotation paths.

    This is a helper function used by
    :func:`vak.prep.prep_helper.make_frame_classification_arrays_from_spectrogram_dataset`.

    Parameters
    ----------
    source_paths : list
        Paths to audio files or array files containing spectrograms.
    labelmap : dict
        Mapping string labels to integer classes predicted by network.
    annot_paths : list
        Paths to annotation files that annotate the spectrograms.
    annot_format : str
        Annotation format of the files in annotation paths.
    crop_dur : float, optional
        Duration to which the entire dataset should be cropped.
        If specified, then ``frame_dur`` must be specified
        so that durations can be measured, and ``labelmap``
        must be specified to ensure that at least one occurrence
        of all classes in dataset are preserved after cropping.
    frame_dur : float, optional
        Duration of a frame, i.e., a single sample in audio
        or a single timebin in a spectrogram.
        Only required if ``crop_dur`` is specified.

    Returns
    -------
    inputs : numpy.NDArray
        An array of inputs to the neural network model,
        concatenated from either audio files
        or spectrograms in array files.
    source_id_vec : numpy.NDArray
        A 1-dimensional vector whose size is equal to the
        width of ``inputs``
    frame_labels : numpy.NDArray or None
    """
    if crop_dur is not None and frame_dur is None:
        raise ValueError("Must provide ``frame_dur`` when specifying ``crop_dur``, "
                         "the duration of a single frame is needed to determine the total duration "
                         "of the dataset and to crop the dataset to the duration specified "
                         "by ``crop_dur``.")

    if crop_dur is not None and labelmap is None:
        raise ValueError("Must provide ``labelmap`` when specifying ``crop_dur``, "
                         "the set of unique class labels for the dataset is needed "
                         "to ensure that set is preserved when cropping the dataset "
                         "to the duration specified by ``crop_dur``.")

    msg = f"Loading data from {len(source_paths)} spectrogram files"
    if annots is not None:
        msg += f" and {len(annots)} annotations"
    logger.info(msg)

    inputs, source_id_vec = [], []
    if annots:
        frame_labels = []
        if crop_dur:
            source_paths, annots = sort_source_paths_and_annots_by_label_freq(source_paths, annots)
        to_do = zip(source_paths, annots)
    else:
        to_do = zip(source_paths, [None] * len(source_paths))

    for source_id, (spect_path, annot) in enumerate(to_do):
        # add to inputs
        d = np.load(spect_path)
        inputs.append(d[spect_key])

        # add to frame labels
        if annot:
            lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
            lbls_frame = transforms.labeled_timebins.from_segments(
                lbls_int,
                annot.seq.onsets_s,
                annot.seq.offsets_s,
                d[timebins_key],
                unlabeled_label=labelmap["unlabeled"],
            )
            frame_labels.append(lbls_frame)

        # add to source_id and source_inds
        n_frames = d["t"].shape[-1]  # number of frames
        source_id_vec.append(
            np.ones((n_frames,)).astype(np.int32) * source_id
        )

    inputs = np.concatenate(inputs, axis=1)
    source_id_vec = np.concatenate(source_id_vec)
    if annots is not None:
        frame_labels = np.concatenate(frame_labels)
    else:
        frame_labels = None

    if crop_dur:
        inputs, source_id_vec, frame_labels = crop_arrays_keep_classes(
            inputs,
            source_id_vec,
            crop_dur,
            frame_dur,
            labelmap,
            frame_labels,
        )

    return inputs, source_id_vec, frame_labels


def make_frame_classification_arrays_from_spectrogram_dataset(
        dataset_df: pd.DataFrame,
        dataset_path: pathlib.Path,
        purpose: str,
        labelmap: dict | None = None,
        spect_key: str = "s",
        timebins_key: str = "t",
) -> None:
    """Makes arrays used by dataset classes
    for frame classification task
    from a dataset of spectrograms
    with annotations.

    Makes one inputs array and one targets array
    per split in the dataframe that represents
    the dataset.

    Parameters
    ----------
    dataset_df : pandas.Dataframe
    dataset_path : str, pathlib.Path
    purpose: str
    labelmap : dict
    spect_key : str
        Key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        Key for accessing vector of time bins in files. Default is 't'.
    """
    dataset_path = pathlib.Path(dataset_path)

    logger.info(f"Will use labelmap: {labelmap}")

    for split in sorted(dataset_df.split.unique()):
        if split == 'None':
            # these are files that didn't get assigned to a split
            continue
        logger.info(f"Processing split: {split}")
        split_dst = dataset_path / split
        logger.info(f"Will save in: {split}")
        split_dst.mkdir(exist_ok=True)

        split_df = dataset_df[dataset_df.split == split]
        source_paths = split_df['spect_path'].values
        if purpose != 'predict':
            annots = common.annotation.from_df(split_df)
        else:
            annots = None

        (inputs,
         source_id_vec,
         frame_labels) = make_frame_classification_arrays_from_source_paths_and_annots(
            source_paths,
            labelmap,
            annots,
            spect_key,
            timebins_key
        )

        logger.info(
            "Saving ``inputs`` vector for frame classification dataset with size "
            f"{round(inputs.nbytes * 1e-6, 2)} MB"
        )
        np.save(split_dst / datasets.frame_classification.constants.INPUT_ARRAY_FILENAME, inputs)
        logger.info(
            "Saving ``source_id`` vector for frame classification dataset with size "
            f"{round(source_id_vec.nbytes * 1e-6, 2)} MB"
        )
        np.save(split_dst / datasets.frame_classification.constants.SOURCE_IDS_ARRAY_FILENAME,
                source_id_vec)
        if purpose != 'predict':
            logger.info(
                "Saving frame labels vector (targets) for frame classification dataset "
                f"with size {round(frame_labels.nbytes * 1e-6, 2)} MB"
            )
            np.save(split_dst / datasets.frame_classification.constants.FRAME_LABELS_ARRAY_FILENAME, frame_labels)
            logger.info(
                "Saving annotations as csv"
            )
            generic_seq = crowsetta.formats.seq.GenericSeq(annots=annots)
            generic_seq.to_file(split_dst / datasets.frame_classification.constants.ANNOTATION_CSV_FILENAME)


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
