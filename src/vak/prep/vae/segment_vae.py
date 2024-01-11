"""Prepare a dataset of segments for a VAE model."""
from __future__ import annotations

import json
import logging
import pathlib

import pandas as pd

from ...common import labels
from ...config.spect_params import SpectParamsConfig
from .. import dataset_df_helper, split
from ..unit_dataset import prep_unit_dataset
from ..parametric_umap import dataset_arrays


logger = logging.getLogger(__name__)


def prep_segment_vae_dataset(
        data_dir: str | pathlib.Path,
        dataset_path: str | pathlib.Path,
        dataset_csv_path: str | pathlib.Path,
        purpose: str,
        audio_format: str | None = None,
        spect_params: SpectParamsConfig | None = None,
        annot_format: str | None = None,
        annot_file: str | pathlib.Path | None = None,
        labelset: set | None = None,
        context_s: float = 0.005,
        max_dur: float | None = None,
        target_shape: tuple[int, int] | None = None,
        train_dur: int | None = None,
        val_dur: int | None = None,
        test_dur: int | None = None,
        train_set_durs: list[float] | None = None,
        num_replicates: int | None = None,
        spect_key: str = "s",
        timebins_key: str = "t",
) -> tuple[pd.DataFrame, tuple[int]]:
    """Prepare a dataset of segments for a VAE model.

    Parameters
    ----------
    data_dir : str, Path
        Path to directory with files from which to make dataset.
    dataset_path
    dataset_csv_path
    purpose : str
        Purpose of the dataset.
        One of {'train', 'eval', 'predict', 'learncurve'}.
        These correspond to commands of the vak command-line interface.
    audio_format : str
        Format of audio files. One of {'wav', 'cbin'}.
        Default is ``None``, but either ``audio_format`` or ``spect_format``
        must be specified.
    spect_params : dict, vak.config.SpectParams
        Parameters for creating spectrograms. Default is ``None``.
    annot_format : str
        Format of annotations. Any format that can be used with the
        :module:`crowsetta` library is valid. Default is ``None``.
    annot_file
    labelset : str, list, set
        Set of unique labels for vocalizations. Strings or integers.
        Default is ``None``. If not ``None``, then files will be skipped
        where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using
        :func:`vak.converters.labelset_to_set`.
        See help for that function for details on how to specify ``labelset``.
    context_s : float
        Number of seconds of "context" around unit to
        add, i.e., time before and after the onset
        and offset respectively. Default is 0.005s,
        5 milliseconds.
    max_dur : float
        Maximum duration for segments.
        If a float value is specified,
        any segment with a duration larger than
        that value (in seconds) will be omitted
        from the dataset. Default is None.
    target_shape : tuple
        Of ints, (target number of frequency bins,
        target number of time bins).
        Spectrograms of units will be reshaped
        by interpolation to have the specified
        number of frequency and time bins.
        The transformation is only applied if both this
        parameter and ``max_dur`` are specified.
        Default is None.
    train_dur : float
        Total duration of training set, in seconds.
        When creating a learning curve,
        training subsets of shorter duration
        will be drawn from this set. Default is None.
    val_dur : float
        Total duration of validation set, in seconds.
        Default is None.
    test_dur : float
        Total duration of test set, in seconds.
        Default is None.
    train_set_durs : list
        of int, durations in seconds of subsets taken from training data
        to create a learning curve, e.g. [5, 10, 15, 20].
    num_replicates : int
        number of times to replicate training for each training set duration
        to better estimate metrics for a training set of that size.
        Each replicate uses a different randomly drawn subset of the training
        data (but of the same duration).
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.

    Returns
    -------

    """
    dataset_df, shape = prep_unit_dataset(
        audio_format,
        dataset_path,
        spect_params,
        data_dir,
        annot_format,
        annot_file,
        labelset,
        context_s,
        max_dur,
        target_shape,
    )
    if dataset_df.empty:
        raise ValueError(
            "Calling `vak.prep.unit_dataset.prep_unit_dataset` "
            "with arguments passed to `vak.core.prep.vae.prep_segment_vae_dataset` "
            "returned an empty dataframe.\n"
            "Please double-check arguments to `vak.core.prep` function."
        )

    # save before (possibly) splitting, just in case duration args are not valid
    # (we can't know until we make dataset)
    dataset_df.to_csv(dataset_csv_path)

    # ---- (possibly) split into train / val / test sets ---------------------------------------------
    # catch case where user specified duration for just training set, raise a helpful error instead of failing silently
    if (purpose == "train" or purpose == "learncurve") and (
            (train_dur is not None and train_dur > 0)
            and (val_dur is None or val_dur == 0)
            and (test_dur is None or val_dur == 0)
    ):
        raise ValueError(
            "A duration was specified for just the training set, "
            "but prep function does not currently support creating a "
            "single split of a specified duration. Either remove the train_dur option from the prep section and "
            "rerun, in which case all data will be included in the training set, or specify values greater than "
            "zero for test_dur (and val_dur, if a validation set will be used)"
        )

    if all(
            [dur is None for dur in (train_dur, val_dur, test_dur)]
    ) or purpose in (
            "eval",
            "predict",
    ):
        # then we're not going to split
        logger.info("Will not split dataset.")
        do_split = False
    else:
        if val_dur is not None and train_dur is None and test_dur is None:
            raise ValueError(
                "cannot specify only val_dur, unclear how to split dataset into training and test sets"
            )
        else:
            logger.info("Will split dataset.")
            do_split = True

    if do_split:
        dataset_df = split.unit_dataframe(
            dataset_df,
            dataset_path,
            labelset=labelset,
            train_dur=train_dur,
            val_dur=val_dur,
            test_dur=test_dur,
        )

    elif (
            do_split is False
    ):  # add a split column, but assign everything to the same 'split'
        # ideally we would just say split=purpose in call to add_split_col, but
        # we have to special case, because "eval" looks for a 'test' split (not an "eval" split)
        if purpose == "eval":
            split_name = (
                "test"  # 'split_name' to avoid name clash with split package
            )
        elif purpose == "predict":
            split_name = "predict"

        dataset_df = dataset_df_helper.add_split_col(
            dataset_df, split=split_name
        )

    # ---- create and save labelmap ------------------------------------------------------------------------------------
    # we do this before creating array files since we need to load the labelmap to make frame label vectors
    if purpose != "predict":
        # TODO: add option to generate predict using existing dataset, so we can get labelmap from it
        labelmap = labels.to_map(labelset, map_unlabeled=False)
        logger.info(
            f"Number of classes in labelmap: {len(labelmap)}",
        )
        # save labelmap in case we need it later
        with (dataset_path / "labelmap.json").open("w") as fp:
            json.dump(labelmap, fp)

    # ---- make arrays that represent final dataset --------------------------------------------------------------------
    dataset_arrays.move_files_into_split_subdirs(
        dataset_df,
        dataset_path,
        purpose,
    )
    #
    # ---- if purpose is learncurve, additionally prep splits for that -----------------------------------------------
    # if purpose == 'learncurve':
    #     dataset_df = make_learncurve_splits_from_dataset_df(
    #         dataset_df,
    #         train_set_durs,
    #         num_replicates,
    #         dataset_path,
    #         labelmap,
    #         audio_format,
    #         spect_key,
    #         timebins_key,
    #     )

    return dataset_df, shape
