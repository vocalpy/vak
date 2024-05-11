from __future__ import annotations

import json
import logging
import pathlib
import warnings

import crowsetta

from ... import datapipes
from ...common import labels
from ...common.converters import expanded_user_path, labelset_to_set
from ...common.logging import config_logging_for_cli, log_version
from ...common.timenow import get_timenow_as_str
from .. import dataset_df_helper, split
from ..unit_dataset import prep_unit_dataset
from . import dataset_arrays

logger = logging.getLogger(__name__)


def prep_parametric_umap_dataset(
    data_dir: str | pathlib.Path,
    purpose: str,
    output_dir: str | pathlib.Path | None = None,
    audio_format: str | None = None,
    spect_params: dict | None = None,
    annot_format: str | None = None,
    annot_file: str | pathlib.Path | None = None,
    labelset: set | None = None,
    context_s: float = 0.015,
    train_dur: int | None = None,
    val_dur: int | None = None,
    test_dur: int | None = None,
    train_set_durs: list[float] | None = None,
    num_replicates: int | None = None,
    spect_key: str = "s",
    timebins_key: str = "t",
):
    """Prepare datasets for neural network models
    that perform a dimensionality reduction task.

    For general information on dataset preparation,
    see the docstring for :func:`vak.prep.prep`.

    Parameters
    ----------
    data_dir : str, Path
        Path to directory with files from which to make dataset.
    purpose : str
        Purpose of the dataset.
        One of {'train', 'eval', 'predict', 'learncurve'}.
        These correspond to commands of the vak command-line interface.
    output_dir : str
        Path to location where data sets should be saved.
        Default is ``None``, in which case it defaults to ``data_dir``.
    audio_format : str
        Format of audio files. One of {'wav', 'cbin'}.
        Default is ``None``, but either ``audio_format`` or ``spect_format``
        must be specified.
    spect_params : dict, vak.config.SpectParams
        Parameters for creating spectrograms. Default is ``None``.
    annot_format : str
        Format of annotations. Any format that can be used with the
        :module:`crowsetta` library is valid. Default is ``None``.
    labelset : str, list, set
        Set of unique labels for vocalizations. Strings or integers.
        Default is ``None``. If not ``None``, then files will be skipped
        where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using
        :func:`vak.converters.labelset_to_set`.
        See help for that function for details on how to specify ``labelset``.
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
    dataset_df : pandas.DataFrame
        That represents a dataset.
    dataset_path : pathlib.Path
        Path to csv saved from ``dataset_df``.
    """
    from .. import constants  # avoid circular import

    # pre-conditions ---------------------------------------------------------------------------------------------------
    if purpose not in constants.VALID_PURPOSES:
        raise ValueError(
            f"purpose must be one of: {constants.VALID_PURPOSES}\n"
            f"Value for purpose was: {purpose}"
        )

    if labelset is not None:
        labelset = labelset_to_set(labelset)

    data_dir = expanded_user_path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Path specified for ``data_dir`` not found: {data_dir}"
        )

    if output_dir:
        output_dir = expanded_user_path(output_dir)
    else:
        output_dir = data_dir

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"Path specified for ``output_dir`` not found: {output_dir}"
        )

    if annot_file is not None:
        annot_file = expanded_user_path(annot_file)
        if not annot_file.exists():
            raise FileNotFoundError(
                f"Path specified for ``annot_file`` not found: {annot_file}"
            )

    if purpose == "predict":
        if labelset is not None:
            warnings.warn(
                "The ``purpose`` argument was set to 'predict`, but a ``labelset`` was provided."
                "This would cause an error because the ``prep_spectrogram_dataset`` section will attempt to "
                "check whether the files in the ``data_dir`` have labels in "
                "``labelset``, even though those files don't have annotation.\n"
                "Setting ``labelset`` to None."
            )
            labelset = None
    else:  # if purpose is not predict
        if labelset is None:
            raise ValueError(
                f"The ``purpose`` argument was set to '{purpose}', but no ``labelset`` was provided."
                "This will cause an error when trying to split the dataset, "
                "e.g. into training and test splits, "
                "or a silent error, e.g. when calculating metrics with an evaluation set. "
                "Please specify a ``labelset`` when calling ``vak.prep.frame_classification.prep`` "
                f"with ``purpose='{purpose}'."
            )

    logger.info(f"Purpose for frame classification dataset: {purpose}")
    # ---- set up directory that will contain dataset, and csv file name -----------------------------------------------
    data_dir_name = data_dir.name
    timenow = get_timenow_as_str()
    dataset_path = (
        output_dir
        / f"{data_dir_name}-vak-dimensionality-reduction-dataset-generated-{timenow}"
    )
    dataset_path.mkdir()

    if annot_file and annot_format == "birdsong-recognition-dataset":
        # we do this normalization / canonicalization after we make dataset_path
        # so that we can put the new annot_file inside of dataset_path, instead of
        # making new files elsewhere on a user's system
        logger.info(
            "The ``annot_format`` argument was set to 'birdsong-recognition-format'; "
            "this format requires the audio files for their sampling rate "
            "to convert onset and offset times of birdsong syllables to seconds."
            "Converting this format to 'generic-seq' now with the times in seconds, "
            "so that the dataset prepared by vak will not require the audio files."
        )
        birdsongrec = crowsetta.formats.seq.BirdsongRec.from_file(annot_file)
        annots = birdsongrec.to_annot()
        # note we point `annot_file` at a new file we're about to make
        annot_file = (
            dataset_path / f"{annot_file.stem}.converted-to-generic-seq.csv"
        )
        # and we remake Annotations here so that annot_path points to this new file, not the birdsong-rec Annotation.xml
        annots = [
            crowsetta.Annotation(
                seq=annot.seq,
                annot_path=annot_file,
                notated_path=annot.notated_path,
            )
            for annot in annots
        ]
        generic_seq = crowsetta.formats.seq.GenericSeq(annots=annots)
        generic_seq.to_file(annot_file)
        # and we now change `annot_format` as well. Both these will get passed to io.prep_spectrogram_dataset
        annot_format = "generic-seq"

    # NOTE we set up logging here (instead of cli) so the prep log is included in the dataset
    config_logging_for_cli(
        log_dst=dataset_path, log_stem="prep", level="INFO", force=True
    )
    log_version(logger)

    dataset_csv_path = dataset_df_helper.get_dataset_csv_path(
        dataset_path, data_dir_name, timenow
    )
    logger.info(f"Will prepare dataset as directory: {dataset_path}")

    # ---- actually make the dataset -----------------------------------------------------------------------------------
    dataset_df, shape = prep_unit_dataset(
        audio_format=audio_format,
        output_dir=dataset_path,
        spect_params=spect_params,
        data_dir=data_dir,
        annot_format=annot_format,
        annot_file=annot_file,
        labelset=labelset,
        context_s=context_s,
    )

    if dataset_df.empty:
        raise ValueError(
            "Calling `vak.prep.unit_dataset.prep_unit_dataset` "
            "with arguments passed to `vak.core.prep.prep_dimensionality_reduction_dataset` "
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
            "A duration specified for just training set, but prep function does not currently support creating a "
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
        labelmap = labels.to_map(labelset, map_background=False)
        logger.info(
            f"Number of classes in labelmap: {len(labelmap)}",
        )
        # save labelmap in case we need it later
        with (dataset_path / "labelmap.json").open("w") as fp:
            json.dump(labelmap, fp)
    else:
        labelmap = None

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

    # ---- save csv file that captures provenance of source data -------------------------------------------------------
    logger.info(f"Saving dataset csv file: {dataset_csv_path}")
    dataset_df.to_csv(
        dataset_csv_path, index=False
    )  # index is False to avoid having "Unnamed: 0" column when loading

    # ---- save metadata -----------------------------------------------------------------------------------------------
    metadata = datapipes.parametric_umap.Metadata(
        dataset_csv_filename=str(dataset_csv_path.name),
        audio_format=audio_format,
        shape=shape,
    )
    metadata.to_json(dataset_path)

    return dataset_df, dataset_path
