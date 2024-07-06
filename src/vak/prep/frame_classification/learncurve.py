"""Functionality to prepare splits of frame classification datasets
to generate a learning curve."""

from __future__ import annotations

import logging
import pathlib
from typing import Sequence

import attrs
import dask.bag as db
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from ... import common, datapipes
from .. import split

logger = logging.getLogger(__name__)


@attrs.define(frozen=True)
class Sample:
    """Dataclass representing one sample
    in a frame classification dataset.

    Used to add paths for arrays from the sample
    to a ``dataset_df``, and to build
    the ``sample_ids`` vector and ``inds_in_sample`` vector
    for the entire dataset."""

    source_id: int = attrs.field()
    sample_id_vec: np.ndarray
    inds_in_sample_vec: np.ndarray


def make_index_vectors_for_each_subset(
    subsets_df: pd.DataFrame,
    dataset_path: str | pathlib.Path,
    input_type: str,
) -> pd.DataFrame:
    r"""Make npy files containing indexing vectors
    for each subset of the training data
    used to generate a learning curve
    with a frame classification dataset.

    This function is basically the same as
    :func:`vak.prep.frame_classification.make_splits.make_splits`,
    *except* that it only makes the indexing vectors
    for each subset of the training data.
    These indexing vectors are needed for each subset
    to properly grab windows from the npy files during training.
    There is no need to remake the npy files themselves though.

    All the indexing vectors for each split are saved
    in the "train" directory split inside ``dataset_path``.

    The indexing vectors are used by
    :class:`vak.datasets.frame_classification.TrainDatapipe`
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
    subset_df : pandas.DataFrame
        A :class:`pandas.DataFrame` representing the training data subsets.
        This DataFrame is created by
        :func:`vak.prep.frame_classification.learncurve.make_subsets_from_dataset_df`,
        and then passed into this function.
        It is created from  a ``pandas.DataFrame``
        returned by :func:`vak.prep.frame_classification.get_or_make_source_files`
        with a ``'split'`` column added.
    dataset_path : pathlib.Path
        Path to directory that represents dataset.
    input_type : str
        The type of input to the neural network model.
        One of {'audio', 'spect'}.

    Returns
    -------
    None
    """
    subsets = [
        subset for subset in sorted(subsets_df.subset.dropna().unique())
    ]
    for subset in subsets:
        logger.info(f"Making indexing vectors for subset: {subset}")
        subset_df = subsets_df[subsets_df.subset == subset].copy()
        frames_paths = subset_df[
            datapipes.frame_classification.constants.FRAMES_PATH_COL_NAME
        ].values

        def _return_index_arrays(
            source_id_path_tup,
        ):
            """Function we use with dask to parallelize.
            Defined in-line so variables are in scope.
            """
            source_id, frames_path = source_id_path_tup

            frames_path = dataset_path / pathlib.Path(frames_path)

            frames = datapipes.frame_classification.helper.load_frames(
                frames_path, input_type
            )

            n_frames = frames.shape[-1]
            sample_id_vec = np.ones((n_frames,)).astype(np.int32) * source_id
            inds_in_sample_vec = np.arange(n_frames)

            return Sample(
                source_id,
                sample_id_vec,
                inds_in_sample_vec,
            )

        # ---- make npy files for this split, parallelized with dask
        # using nested function just defined
        source_id_frames_path_tups = [
            (source_id, frames_path)
            for source_id, frames_path in enumerate(frames_paths)
        ]

        source_id_frames_path_bag = db.from_sequence(
            source_id_frames_path_tups
        )
        with ProgressBar():
            samples = list(source_id_frames_path_bag.map(_return_index_arrays))
        samples = sorted(samples, key=lambda sample: sample.source_id)

        # ---- save indexing vectors in train directory
        sample_id_vec = np.concatenate(
            list(sample.sample_id_vec for sample in samples)
        )
        np.save(
            dataset_path
            / "train"
            / datapipes.frame_classification.helper.sample_ids_array_filename_for_subset(
                subset
            ),
            sample_id_vec,
        )
        inds_in_sample_vec = np.concatenate(
            list(sample.inds_in_sample_vec for sample in samples)
        )
        np.save(
            dataset_path
            / "train"
            / datapipes.frame_classification.helper.inds_in_sample_array_filename_for_subset(
                subset
            ),
            inds_in_sample_vec,
        )


def make_subsets_from_dataset_df(
    dataset_df: pd.DataFrame,
    input_type: str,
    train_set_durs: Sequence[float],
    num_replicates: int,
    dataset_path: pathlib.Path,
    labelmap: dict,
    background_label: str = common.constants.DEFAULT_BACKGROUND_LABEL,
) -> pd.DataFrame:
    """Make subsets of the training data split for a learning curve.

     Makes subsets given a dataframe representing the entire dataset,
     with one subset for each combination of (training set duration,
     replicate number). Each subset is randomly drawn
     from the total training split.

     Uses :func:`vak.prep.split.frame_classification_dataframe` to make
     subsets of the training data from ``dataset_df``.

     A new column will be added to the dataframe, `'subset'`,
     and additional rows for each subset.
     The dataframe is returned with these subsets added.
     (The `'split'` for these rows will still be `'train'`.)
     Additionally, a separate set of indexing vectors
     will be made for each subset, using
     :func:`vak.prep.frame_classification.learncurve.make_index_vectors_for_each_subset`.

    .. code-block:: console

       032312-vak-frame-classification-dataset-generated-231005_121809
       ├── 032312_prep_231005_121809.csv
       ├── labelmap.json
       ├── metadata.json
       ├── prep_231005_121809.log
       ├── TweetyNet_learncurve_audio_cbin_annot_notmat.toml
       ├── train
           ├── gy6or6_baseline_230312_0808.138.cbin.spect.frame_labels.npy
           ├── gy6or6_baseline_230312_0808.138.cbin.spect.frames.npy
           ├── gy6or6_baseline_230312_0809.141.cbin.spect.frame_labels.npy
           ├── gy6or6_baseline_230312_0809.141.cbin.spect.frames.npy
           ├── gy6or6_baseline_230312_0813.163.cbin.spect.frame_labels.npy
           ├── gy6or6_baseline_230312_0813.163.cbin.spect.frames.npy
           ├── gy6or6_baseline_230312_0816.179.cbin.spect.frame_labels.npy
           ├── gy6or6_baseline_230312_0816.179.cbin.spect.frames.npy
           ├── gy6or6_baseline_230312_0820.196.cbin.spect.frame_labels.npy
           ├── gy6or6_baseline_230312_0820.196.cbin.spect.frames.npy
           ├── inds_in_sample.npy
           ├── inds_in_sample-train-dur-4.0-replicate-1.npy
           ├── inds_in_sample-train-dur-4.0-replicate-2.npy
           ├── inds_in_sample-train-dur-6.0-replicate-1.npy
           ├── inds_in_sample-train-dur-6.0-replicate-2.npy
           ├── sample_ids.npy
           ├── sample_ids-train-dur-4.0-replicate-1.npy
           ├── sample_ids-train-dur-4.0-replicate-2.npy
           ├── sample_ids-train-dur-6.0-replicate-1.npy
           └── sample_ids-train-dur-6.0-replicate-2.npy
       ...

     Parameters
     ----------
     dataset_df : pandas.DataFrame
         Dataframe representing a dataset for frame classification models.
         It is returned by
         :func:`vak.prep.frame_classification.get_or_make_source_files`,
         and has a ``'split'`` column added.
     train_set_durs : list
         Durations in seconds of subsets taken from training data
         to create a learning curve, e.g., `[5., 10., 15., 20.]`.
     num_replicates : int
         number of times to replicate training for each training set duration
         to better estimate metrics for a training set of that size.
         Each replicate uses a different randomly drawn subset of the training
         data (but of the same duration).
     dataset_path : str, pathlib.Path
         Directory where splits will be saved.
     input_type : str
         The type of input to the neural network model.
         One of {'audio', 'spect'}.
    background_label: str, optional
        The string label applied to segments belonging to the
        background class.
        Default is
        :const:`vak.common.constants.DEFAULT_BACKGROUND_LABEL`.

     Returns
     -------
     dataset_df_out : pandas.DataFrame
         A pandas.DataFrame that has the original splits
         from ``dataset_df``, as well as the additional subsets
         of the training data added, along with additional
         columns, ``'subset', 'train_dur', 'replicate_num'``,
         that are used by :mod:`vak`.
         Other functions like :func:`vak.learncurve.learncurve`
         specify a specific subset of the training data
         by getting the subset name with the function
         :func:`vak.common.learncurve.get_train_dur_replicate_split_name`,
         and then filtering ``dataset_df_out`` with that name
         using the 'subset' column.
    """
    dataset_path = pathlib.Path(dataset_path)

    # get just train split, to pass to split.dataframe
    # so we don't end up with other splits in the training set
    train_split_df = dataset_df[dataset_df["split"] == "train"].copy()
    labelset = set([k for k in labelmap.keys() if k != background_label])

    # will concat after loop, then use ``csv_path`` to replace
    # original dataset df with this one
    subsets_df = []
    for train_dur in train_set_durs:
        logger.info(
            f"Subsetting training set for training set of duration: {train_dur}",
        )
        for replicate_num in range(1, num_replicates + 1):
            train_dur_replicate_subset_name = (
                common.learncurve.get_train_dur_replicate_subset_name(
                    train_dur, replicate_num
                )
            )

            train_dur_replicate_df = split.frame_classification_dataframe(
                # copy to avoid mutating original train_split_df
                train_split_df.copy(),
                dataset_path,
                train_dur=train_dur,
                labelset=labelset,
            )
            # remove rows where split set to 'None'
            train_dur_replicate_df = train_dur_replicate_df[
                train_dur_replicate_df.split == "train"
            ]
            # next line, make split name in csv match the split name used for directory in dataset dir
            train_dur_replicate_df["subset"] = train_dur_replicate_subset_name
            train_dur_replicate_df["train_dur"] = train_dur
            train_dur_replicate_df["replicate_num"] = replicate_num
            subsets_df.append(train_dur_replicate_df)

    subsets_df = pd.concat(subsets_df)

    make_index_vectors_for_each_subset(
        subsets_df,
        dataset_path,
        input_type,
    )

    # keep the same validation, test, and total train sets by concatenating them with the train subsets
    dataset_df["subset"] = None  # add column but have it be empty
    dataset_df = pd.concat((subsets_df, dataset_df))
    # We reset the entire index across all splits, instead of repeating indices,
    # and we set drop=False because we don't want to add a new column 'index' or 'level_0'.
    # Need to do this again after calling `make_npy_files_for_each_split` since we just
    # did `pd.concat` with the original dataframe
    dataset_df = dataset_df.reset_index(drop=True)
    return dataset_df
