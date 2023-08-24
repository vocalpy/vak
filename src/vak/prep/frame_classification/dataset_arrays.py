"""Helper functions for frame classification dataset prep."""
from __future__ import annotations

import collections
import copy
import logging
import pathlib

import attrs
import crowsetta
import dask.bag as db
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from ... import common, datasets, transforms
from .. import constants as prep_constants

logger = logging.getLogger(__name__)


def argsort_by_label_freq(annots: list[crowsetta.Annotation]) -> list[int]:
    """Returns indices to sort a list of annotations
     in order of more frequently appearing labels,
     i.e., the first annotation will have the label
     that appears least frequently and the last annotation
     will have the label that appears most frequently.

    Used to sort a dataframe representing a dataset of annotated audio
    or spectrograms before cropping that dataset to a specified duration,
    so that it's less likely that cropping will remove all occurrences
    of any label class from the total dataset.

     Parameters
     ----------
     annots: list
         List of :class:`crowsetta.Annotation` instances.

     Returns
     -------
     sort_inds: list
         Integer values to sort ``annots``.
    """
    all_labels = [lbl for annot in annots for lbl in annot.seq.labels]
    label_counts = collections.Counter(all_labels)

    sort_inds = []
    # make indices ahead of time so they stay constant as we remove things from the list
    ind_annot_tuples = list(enumerate(copy.deepcopy(annots)))
    for label, _ in reversed(label_counts.most_common()):
        # next line, [:] to make a temporary copy to avoid remove bug
        for ind_annot_tuple in ind_annot_tuples[:]:
            ind, annot = ind_annot_tuple
            if label in annot.seq.labels.tolist():
                sort_inds.append(ind)
                ind_annot_tuples.remove(ind_annot_tuple)

    # make sure we got all source_paths + annots
    if len(ind_annot_tuples) > 0:
        for ind_annot_tuple in ind_annot_tuples:
            ind, annot = ind_annot_tuple
            sort_inds.append(ind)
            ind_annot_tuples.remove(ind_annot_tuple)

    if len(ind_annot_tuples) > 0:
        raise ValueError(
            "Not all ``annots`` were used in sorting."
            f"Left over (with indices from list): {ind_annot_tuples}"
        )

    if not (sorted(sort_inds) == list(range(len(annots)))):
        raise ValueError(
            "sorted(sort_inds) does not equal range(len(annots)):"
            f"sort_inds: {sort_inds}\nrange(len(annots)): {list(range(len(annots)))}"
        )

    return sort_inds


@attrs.define(frozen=True)
class Sample:
    """Dataclass representing one sample
    in a frame classification dataset.

    Used to add paths for arrays from the sample
    to a ``dataset_df``, and to build
    the ``sample_ids`` vector and ``inds_in_sample`` vector
    for the entire dataset."""

    source_id: int = attrs.field()
    frame_npy_path: str
    frame_labels_npy_path: str
    sample_id_vec: np.ndarray
    inds_in_sample_vec: np.ndarray


def make_npy_files_for_each_split(
    dataset_df: pd.DataFrame,
    dataset_path: str | pathlib.Path,
    input_type: str,
    purpose: str,
    labelmap: dict,
    audio_format: str,
    spect_key: str = "s",
    timebins_key: str = "t",
) -> pd.DataFrame:
    r"""Make npy files containing arrays
    for each split of a frame classification dataset.

    All the npy files for each split are saved
    in a new directory inside ``dataset_path``
    that has the same name as the split.
    E.g., the ``train`` directory inside ``dataset_path``
    would have all the files for every row in ``dataset_df``
    for which ``dataset_df['split'] == 'train'``.

    The function creates two npy files for each row in ``dataset_df``.
    One has the extension '.frames.npy` and contains the input
    to the frame classification model. The other has the extension
    '.frame_labels.npy', and contains a vector
    where each element is the target label that
    the network should predict for the corresponding frame.
    Taken together, these two files are the data
    for each sample :math:`(x, y)` in the dataset,
    where :math:`x_t` is the frames and :math:`y_t` is the frame labels.

    This function also creates two additional npy files for each split.
    These npy files are "indexing" vectors that
    are used by :class:`vak.datasets.frame_classification.WindowDataset`
    and :class:`vak.datasets.frame_classification.FramesDataset`.
    These vectors make it possible to work with files,
    to avoid loading the entire dataset into memory,
    and to avoid working with memory-mapped arrays.
    The first is the ``sample_ids`` vector,
    that represents the "ID" of any sample :math:`(x, y)` in the dataset.
    We use these IDs to load the array files corresponding to the samples.
    For a dataset with :math:`m` samples, this will be an array of length :math:`T`,
    the total number of frames across all samples,
    with elements :math:`i \in (0, 1, ..., m - 1)`
    indicating which frames correspond to which sample :math:`m_i`:
    :math:`(0, 0, 0, ..., 1, 1, ..., m - 1, m -1)`.
    The second vector is the ``inds_in_sample`` vector.
    This vector is the same length as ``sample_ids``, but its values represent
    the indices of frames within each sample :math:`x_t`.
    For a data set with :math:`T` total frames across all samples,
    where :math:`t_i` indicates the number of frames in each :math:`x_i`,
    this vector will look like :math:`(0, 1, ..., t_0, 0, 1, ..., t_1, ... t_m)`.

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
    input_type : str
        The type of input to the neural network model.
        One of {'audio', 'spect'}.
    purpose: str
        A string indicating what the dataset will be used for.
        One of {'train', 'eval', 'predict', 'learncurve'}.
        Determined by :func:`vak.core.prep.prep`
        using the TOML configuration file.
    labelmap : dict
        A :class:`dict` that maps a set of human-readable
        string labels to the integer classes predicted by a neural
        network model. As returned by :func:`vak.labels.to_map`.
    audio_format : str
        A :class:`string` representing the format of audio files.
        One of :constant:`vak.common.constants.VALID_AUDIO_FORMATS`.
    spect_key : str
        Key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        Key for accessing vector of time bins in files. Default is 't'.

    Returns
    -------
    dataset_df_out : pandas.DataFrame
        The ``dataset_df`` with splits sorted by increasing frequency
        of labels (see :func:`~vak.prep.frame_classification.dataset_arrays`),
        and with columns added containing the npy files for each row.
    """
    if input_type not in prep_constants.INPUT_TYPES:
        raise ValueError(
            f"``input_type`` must be one of: {prep_constants.INPUT_TYPES}\n"
            f"Value for ``input_type`` was: {input_type}"
        )

    dataset_df_out = []
    splits = [
        split
        for split in sorted(dataset_df.split.dropna().unique())
        if split != "None"
    ]
    for split in splits:
        split_subdir = dataset_path / split
        split_subdir.mkdir()

        split_df = dataset_df[dataset_df.split == split].copy()

        if purpose != "predict":
            annots = common.annotation.from_df(split_df)
        else:
            annots = None

        if annots:
            sort_inds = argsort_by_label_freq(annots)
            split_df["sort_inds"] = sort_inds
            split_df = (
                split_df.sort_values(by="sort_inds")
                .drop(columns="sort_inds")
            )

        if input_type == "audio":
            source_paths = split_df["audio_path"].values
        elif input_type == "spect":
            source_paths = split_df["spect_path"].values
        else:
            raise ValueError(f"Invalid ``input_type``: {input_type}")
        # do this *again* after sorting the dataframe
        if purpose != "predict":
            annots = common.annotation.from_df(split_df)
        else:
            annots = None

        def _save_dataset_arrays_and_return_index_arrays(
            source_id_path_annot_tup,
        ):
            """Function we use with dask to parallelize

            Defined in-line so variables are in scope
            """
            source_id, source_path, annot = source_id_path_annot_tup
            source_path = pathlib.Path(source_path)

            if input_type == "audio":
                frames, samplefreq = common.constants.AUDIO_FORMAT_FUNC_MAP[
                    audio_format
                ](source_path)
                if (
                    audio_format == "cbin"
                ):  # convert to ~wav, from int16 to float64
                    frames = frames.astype(np.float64) / 32768.0
                if annot:
                    frame_times = np.arange(frames.shape[-1]) / samplefreq
            elif input_type == "spect":
                spect_dict = np.load(source_path)
                frames = spect_dict[spect_key]
                if annot:
                    frame_times = spect_dict[timebins_key]
            frames_npy_path = split_subdir / (
                source_path.stem
                + datasets.frame_classification.constants.FRAMES_ARRAY_EXT
            )
            np.save(frames_npy_path, frames)
            frames_npy_path = str(
                # make sure we save path in csv as relative to dataset root
                frames_npy_path.relative_to(dataset_path)
            )

            n_frames = frames.shape[-1]
            sample_id_vec = np.ones((n_frames,)).astype(np.int32) * source_id
            inds_in_sample_vec = np.arange(n_frames)

            # add to frame labels
            if annot:
                lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
                frame_labels = transforms.frame_labels.from_segments(
                    lbls_int,
                    annot.seq.onsets_s,
                    annot.seq.offsets_s,
                    frame_times,
                    unlabeled_label=labelmap["unlabeled"],
                )
                frame_labels_npy_path = split_subdir / (
                    source_path.stem
                    + datasets.frame_classification.constants.FRAME_LABELS_EXT
                )
                np.save(frame_labels_npy_path, frame_labels)
                frame_labels_npy_path = str(
                    # make sure we save path in csv as relative to dataset root
                    frame_labels_npy_path.relative_to(dataset_path)
                )
            else:
                frame_labels_npy_path = None

            return Sample(
                source_id,
                frames_npy_path,
                frame_labels_npy_path,
                sample_id_vec,
                inds_in_sample_vec,
            )

        # ---- make npy files for this split, parallelized with dask
        # using nested function just defined
        if annots:
            source_path_annot_tups = [
                (source_id, source_path, annot)
                for source_id, (source_path, annot) in enumerate(
                    zip(source_paths, annots)
                )
            ]
        else:
            source_path_annot_tups = [
                (source_id, source_path, None)
                for source_id, source_path in enumerate(source_paths)
            ]

        source_path_annot_bag = db.from_sequence(source_path_annot_tups)
        with ProgressBar():
            samples = list(
                source_path_annot_bag.map(
                    _save_dataset_arrays_and_return_index_arrays
                )
            )
        samples = sorted(samples, key=lambda sample: sample.source_id)

        # ---- save indexing vectors in split directory
        sample_id_vec = np.concatenate(
            list(sample.sample_id_vec for sample in samples)
        )
        np.save(
            split_subdir
            / datasets.frame_classification.constants.SAMPLE_IDS_ARRAY_FILENAME,
            sample_id_vec,
        )
        inds_in_sample_vec = np.concatenate(
            list(sample.inds_in_sample_vec for sample in samples)
        )
        np.save(
            split_subdir
            / datasets.frame_classification.constants.INDS_IN_SAMPLE_ARRAY_FILENAME,
            inds_in_sample_vec,
        )

        frame_npy_paths = [str(sample.frame_npy_path) for sample in samples]
        split_df[
            datasets.frame_classification.constants.FRAMES_NPY_PATH_COL_NAME
        ] = frame_npy_paths

        frame_labels_npy_paths = [
            str(sample.frame_labels_npy_path) for sample in samples
        ]
        split_df[
            datasets.frame_classification.constants.FRAME_LABELS_NPY_PATH_COL_NAME
        ] = frame_labels_npy_paths
        dataset_df_out.append(split_df)

    # we reset the entire index across all splits, instead of repeating indices,
    # and we set drop=False because we don't want to add a new column 'index' or 'level_0'
    dataset_df_out = pd.concat(dataset_df_out).reset_index(drop=True)
    return dataset_df_out
