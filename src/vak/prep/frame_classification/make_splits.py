"""Helper functions for frame classification dataset prep."""

from __future__ import annotations

import collections
import copy
import logging
import pathlib
import shutil

import attrs
import crowsetta
import dask.bag as db
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from ... import common, datapipes, transforms
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
    for the entire dataset.

    Attributes
    ----------
    source_id : int
        Integer ID number used for sorting.
    frames_path : str
        The path to the input to the model
        :math:`x` after it has been moved,
        copied, or created from a ``source_path``.
        Path will be written relative to ``dataset_path``.
        We preserve the original paths as metadata,
        and consider the files in the split to contain
        frames, regardless of the source domain
        of the data.
    frame_labels_npy_path : str
        Path to frame labels,
        relative to ``dataset_path``.
    sample_id_vec : numpy.ndarray
        Sample ID vector for this sample.
    inds_in_sample_vec : numpy.ndarray
        Indices within sample.
    """

    source_id: int = attrs.field()
    source_path: str
    frame_labels_npy_path: str
    sample_id_vec: np.ndarray
    inds_in_sample_vec: np.ndarray


def make_splits(
    dataset_df: pd.DataFrame,
    dataset_path: str | pathlib.Path,
    input_type: str,
    purpose: str,
    labelmap: dict,
    audio_format: str | None = None,
    spect_key: str = "s",
    timebins_key: str = "t",
    freqbins_key: str = "f",
    background_label: str = common.constants.DEFAULT_BACKGROUND_LABEL,
) -> pd.DataFrame:
    r"""Make each split of a frame classification dataset.

    This function takes a :class:`pandas.Dataframe` returned by
    :func:`vak.prep.spectrogram_dataset.prep_spectrogram_dataset`
    or :func:`vak.prep.audio_dataset.prep_audio_dataset`,
    after it has been assigned a `'split'` column,
    and then copies, moves, or generates the required files
    as appropriate for each split.

    For each unique `'split'` in the :class:`pandas.Dataframe`,
    a directory is made inside ``dataset_path``.
    At a high level, all files needed for working with that split
    will be in that directory
    E.g., the ``train`` directory inside ``dataset_path``
    would have all the files for every row in ``dataset_df``
    for which ``dataset_df['split'] == 'train'``.

    The inputs to the neural network model
    are moved or copied into the split directory,
    or generated if necessary.
    If the ``input_type`` is `'audio'`,
    then the audio files are copied from their original directory.
    If the ``input_type`` is `'spect'`,
    and the spectrogram files are already
    in ``dataset_path``, they are moved into the split directory
    (under the assumption they were generated
    by ``vak.prep.spectrogram_dataset.audio_helper``).
    If they are npz files, but they are not in ``dataset_path``,
    then they are validated to make sure they have the appropriate keys,
    and then copied into the split directory.
    This could be the case if the files were generated
    by another program.
    If they are mat files, they will be converted to npz
    with the default keys for arrays,
    and then saved in a new npz file in the split directory.
    This step is required so that all dataset
    prepared by :mod:`vak` are in a "normalized" or
    "canonicalized" format.

    In addition to copying or moving the audio or spectrogram
    files that are inputs to the neural network model,
    other npy files are made for each split
    and saved in the corresponding directory.
    This function creates one npy file for each row in ``dataset_df``.
    It has the extension '.frame_labels.npy', and contains a vector
    where each element is the target label that
    the network should predict for the corresponding frame.
    Taken together, the audio or spectrogram file in each row
    along with its corresponding frame labels are the data
    for each sample :math:`(x, y)` in the dataset,
    where :math:`x_t` supplies the "frames", and :math:`y_t` is the frame labels.

    This function also creates two additional npy files for each split.
    These npy files are "indexing" vectors that
    are used by :class:`vak.datasets.frame_classification.TrainDatapipe`
    and :class:`vak.datasets.frame_classification.InferDatapipe`.
    These vectors make it possible to work with files,
    to avoid loading the entire dataset into memory,
    and to avoid working with memory-mapped arrays.
    The first is the ``sample_ids`` vector,
    that represents the "ID" of any sample :math:`(x, y)` in the split.
    We use these IDs to load the array files corresponding to the samples.
    For a split with :math:`m` samples, this will be an array of length :math:`T`,
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
        One of :const:`vak.common.constants.VALID_AUDIO_FORMATS`.
    spect_key : str
        Key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        Key for accessing vector of time bins in files. Default is 't'.
    freqbins_key : str
        key for accessing vector of frequency bins in files. Default is 'f'.
    background_label: str, optional
        The string label applied to segments belonging to the
        background class.
        Default is
        :const:`vak.common.constants.DEFAULT_BACKGROUND_LABEL`.

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

    if input_type == "audio" and audio_format is None:
        raise ValueError(
            "Value for `input_type` was 'audio' but `audio_format` is None. "
            "Please specify the audio format."
        )

    dataset_df_out = []
    splits = [
        split
        for split in sorted(dataset_df.split.dropna().unique())
        if split != "None"
    ]
    for split in splits:
        logger.info(f"Making split for dataset: {split}")
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
            split_df = split_df.sort_values(by="sort_inds").drop(
                columns="sort_inds"
            )

        if input_type == "audio":
            source_paths = split_df["audio_path"].values
        elif input_type == "spect":
            source_paths = split_df["spect_path"].values
        else:
            raise ValueError(f"Invalid ``input_type``: {input_type}")
        source_paths = [
            pathlib.Path(source_path) for source_path in source_paths
        ]

        # we get annots again, *after* sorting the dataframe
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

            if input_type == "audio":
                # we always copy audio to the split directory, to avoid damaging source data
                frames_path = shutil.copy(source_path, split_subdir)
                # after copying, we load frames to compute frame labels
                frames, samplefreq = common.constants.AUDIO_FORMAT_FUNC_MAP[
                    audio_format
                ](source_path)
                if (
                    audio_format == "cbin"
                ):  # convert to ~wav, from int16 to float64damage
                    frames = frames.astype(np.float64) / 32768.0
                if annot:
                    frame_times = np.arange(frames.shape[-1]) / samplefreq
            elif input_type == "spect":
                if source_path.suffix.endswith("mat"):
                    spect_dict = common.files.spect.load(source_path, "mat")
                    # convert to .npz and save in spect_output_dir
                    spect_dict_npz = {
                        "s": spect_dict[spect_key],
                        "t": spect_dict[timebins_key],
                        "f": spect_dict[freqbins_key],
                    }
                    frames_path = split_subdir / (source_path.stem + ".npz")
                    np.savez(frames_path, **spect_dict_npz)
                elif source_path.suffix.endswith("npz"):
                    spect_dict = common.files.spect.load(source_path, "npz")
                    if source_path.is_relative_to(dataset_path):
                        # it's already in dataset_path, we just move it into the split
                        frames_path = shutil.move(source_path, split_subdir)
                    else:
                        # it's somewhere else we copy it to be safe
                        if not all(
                            [key in spect_dict for key in ("s", "t", "f")]
                        ):
                            raise ValueError(
                                f"The following spectrogram file did not have valid keys: {source_path}\n."
                                f"All npz files should have keys 's', 't', 'f' corresponding to the spectrogram,"
                                f"the frequencies vector, and the time vector."
                            )
                        frames_path = shutil.copy(source_path, split_subdir)
                frames = spect_dict[spect_key]
                if annot:
                    frame_times = spect_dict[timebins_key]

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
                    background_label=labelmap[background_label],
                )
                frame_labels_npy_path = split_subdir / (
                    source_path.stem
                    + datapipes.frame_classification.constants.MULTI_FRAME_LABELS_EXT
                )
                np.save(frame_labels_npy_path, frame_labels)
                frame_labels_npy_path = str(
                    # make sure we save path in csv as relative to dataset root
                    frame_labels_npy_path.relative_to(dataset_path)
                )
            else:
                frame_labels_npy_path = None

            # Rewrite ``frames_path`` as relative to root
            # because all functions and classes downstream expect this
            frames_path = pathlib.Path(frames_path).relative_to(dataset_path)

            return Sample(
                source_id,
                frames_path,
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
            / datapipes.frame_classification.constants.SAMPLE_IDS_ARRAY_FILENAME,
            sample_id_vec,
        )
        inds_in_sample_vec = np.concatenate(
            list(sample.inds_in_sample_vec for sample in samples)
        )
        np.save(
            split_subdir
            / datapipes.frame_classification.constants.INDS_IN_SAMPLE_ARRAY_FILENAME,
            inds_in_sample_vec,
        )

        # We convert `frames_paths` back to string
        # (just in case they are pathlib.Paths) before adding back to dataframe.
        # Note that these are all in split dirs, written relative to ``dataset_path``.
        frames_paths = [str(sample.source_path) for sample in samples]
        split_df[
            datapipes.frame_classification.constants.FRAMES_PATH_COL_NAME
        ] = frames_paths

        frame_labels_npy_paths = [
            (
                sample.frame_labels_npy_path
                if isinstance(sample.frame_labels_npy_path, str)
                else None
            )
            for sample in samples
        ]
        split_df[
            datapipes.frame_classification.constants.MULTI_FRAME_LABELS_PATH_COL_NAME
        ] = frame_labels_npy_paths
        dataset_df_out.append(split_df)

    # ---- clean up
    # Remove any spect npz files that were *not* added to a split
    spect_npz_files_not_in_split = sorted(
        dataset_path.glob(f"*{common.constants.SPECT_NPZ_EXTENSION}")
    )
    if len(spect_npz_files_not_in_split) > 0:
        for spect_npz_file in spect_npz_files_not_in_split:
            spect_npz_file.unlink()

    # we reset the entire index across all splits, instead of repeating indices,
    # and we set drop=False because we don't want to add a new column 'index' or 'level_0'
    dataset_df_out = pd.concat(dataset_df_out).reset_index(drop=True)
    return dataset_df_out
