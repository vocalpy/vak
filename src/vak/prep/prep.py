import json
import logging
import pathlib
import warnings

import crowsetta.formats.seq

from . import prep_helper, split
from .learncurve import make_learncurve_splits_from_dataset_df
from .spectrogram_dataset.prep import prep_spectrogram_dataset

from .. import datasets
from ..common import labels
from ..common.converters import expanded_user_path, labelset_to_set
from ..common.logging import config_logging_for_cli, log_version
from ..common.timenow import get_timenow_as_str
from ..datasets.metadata import Metadata



__all__ = [
    'prep'
]


logger = logging.getLogger(__name__)



def prep(
    data_dir: str | pathlib.Path,
    purpose: str,
    output_dir: str | pathlib.Path | None = None,
    audio_format: str | None = None,
    spect_format: str | None = None,
    spect_params: dict | None = None,
    annot_format: str | None = None,
    annot_file: str | pathlib.Path | None = None,
    labelset: set | None = None,
    audio_dask_bag_kwargs: dict | None = None,
    train_dur: int | None = None,
    val_dur: int | None =None,
    test_dur: int | None = None,
    train_set_durs: list[float] | None = None,
    num_replicates: int | None = None,
    window_size: int | None = None,
    spect_key: str = "s",
    timebins_key: str = "t",
):
    """Prepare datasets for use with neural network models.

    Datasets are used to train and evaluate neural networks.
    The function also prepares datasets to generate predictions
    with trained neural networks, and to train a series of models
    with varying sizes of dataset so that performance
    can be evaluated as a function of dataset size,
    with a learning curve.

    When :func:`vak.core.prep` runs, it builds the dataset in
    ``data_dir`` containing the data itself as
    well as additional metadata.

    This is a high-level function that prepares datasets to be used by other
    high-level functions like :func:`vak.core.train`,
    :func:`vak.core.predict`, and :func:`vak.core.learncurve`.

    It can also split a dataset into training, validation, and test sets,
    e.g. for benchmarking different neural network architectures.
    If the ``purpose`` argument is set to 'train' or 'learncurve',
    and/or the duration of either the training or test set is provided,
    then the function attempts to split the dataset into training and test sets.
    A duration can also be specified for a validation set
    (used to measure performance during training).
    In these cases, the 'split' column in the .csv
    identifies which files (rows) belong to the training, test, and
    validation sets created from that Dataset.

    If the ``purpose`` is set to 'predict' or 'eval',
    or no durations for any of the training sets are specified,
    then the function assumes all the vocalizations constitute a single
    dataset, and for all rows the 'split' columns for that dataset
    will be 'predict' or 'test' (respectively).

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
    spect_format : str
        Format of files containing spectrograms as 2-d matrices. One of {'mat', 'npz'}.
        Default is None, but either audio_format or spect_format must be specified.
    spect_params : dict, vak.config.SpectParams
        Parameters for creating spectrograms. Default is ``None``.
    annot_format : str
        Format of annotations. Any format that can be used with the
        :module:`crowsetta` library is valid. Default is ``None``.
    annot_file : str
        Path to a single annotation file. Default is ``None``.
        Used when a single file contains annotates multiple audio
        or spectrogram files.
    labelset : str, list, set
        Set of unique labels for vocalizations. Strings or integers.
        Default is ``None``. If not ``None``, then files will be skipped
        where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using
        :func:`vak.converters.labelset_to_set`.
        See help for that function for details on how to specify ``labelset``.
    audio_dask_bag_kwargs : dict
        Keyword arguments used when calling :func:`dask.bag.from_sequence`
        inside :func:`vak.io.audio`, where it is used to parallelize
        the conversion of audio files into spectrograms.
        Option should be specified in config.toml file as an inline table,
        e.g., ``audio_dask_bag_kwargs = { npartitions = 20 }``.
        Allows for finer-grained control
        when needed to process files of different sizes.
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
    window_size : int
        Size of windows taken from spectrograms, in number of time bins,
        shown to neural networks
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
    # pre-conditions ---------------------------------------------------------------------------------------------------
    if purpose not in prep_helper.VALID_PURPOSES:
        raise ValueError(
            f"purpose must be one of: {prep_helper.VALID_PURPOSES}\n"
            f"Value for purpose was: {purpose}"
        )

    if audio_format is None and spect_format is None:
        raise ValueError("Must specify either audio_format or spect_format")

    if audio_format and spect_format:
        raise ValueError(
            "Cannot specify both audio_format and spect_format, "
            "unclear whether to create spectrograms from audio files or "
            "use already-generated spectrograms from array files"
        )

    if labelset is not None:
        labelset = labelset_to_set(labelset)

    data_dir = expanded_user_path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir not found: {data_dir}")

    if output_dir:
        output_dir = expanded_user_path(output_dir)
    else:
        output_dir = data_dir

    if not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir not found: {output_dir}")

    if annot_file is not None:
        annot_file = expanded_user_path(annot_file)
        if not annot_file.exists():
            raise FileNotFoundError(
                f'annot_file not found: {annot_file}'
            )

    if purpose == "predict":
        if labelset is not None:
            warnings.warn(
                ".toml config file has a 'predict' section, but a labelset was provided."
                "This would cause an error because the prep_spectrogram_dataset section will attempt to "
                f"check whether the files in the data_dir ({data_dir}) have labels in "
                "labelset, even though those files don't have annotation.\n"
                "Setting labelset to None."
            )
            labelset = None
    else:  # if purpose is not predict
        if labelset is None:
            raise ValueError(
                f".toml config file has a '{purpose}' section, but no labelset was provided."
                "This will cause an error when trying to split the dataset, "
                "e.g. into training and test splits, "
                "or a silent error, e.g. when calculating metrics with an evaluation set. "
                "Please add a 'labelset' option to the [PREP] section of the .toml config file."
            )

    logger.info(f"purpose for dataset: {purpose}")
    # ---- set up directory that will contain dataset, and csv file name -----------------------------------------------
    data_dir_name = data_dir.name
    timenow = get_timenow_as_str()
    # TODO: add 'dataset_name' parameter that overrides this default
    # TODO: different default?
    dataset_path = output_dir / f'{data_dir_name}-vak-dataset-generated-{timenow}'
    dataset_path.mkdir()

    if annot_file and annot_format == 'birdsong-recognition-dataset':
        # we do this normalization / canonicalization after we make dataset_path
        # so that we can put the new annot_file inside of dataset_path, instead of
        # making new files elsewhere on a user's system
        logger.info("The `annot_format` argument was set to 'birdsong-recognition-format'; "
                    "this format requires the audio files for their sampling rate "
                    "to convert onset and offset times of birdsong syllables to seconds."
                    "Converting this format to 'generic-seq' now with the times in seconds, "
                    "so that the dataset prepared by vak will not require the audio files.")
        birdsongrec = crowsetta.formats.seq.BirdsongRec.from_file(annot_file)
        annots = birdsongrec.to_annot()
        # note we point `annot_file` at a new file we're about to make
        annot_file = dataset_path / f'{annot_file.stem}.converted-to-generic-seq.csv'
        # and we remake Annotations here so that annot_path points to this new file, not the birdsong-rec Annotation.xml
        annots = [
            crowsetta.Annotation(seq=annot.seq, annot_path=annot_file, notated_path=annot.notated_path)
            for annot in annots
        ]
        generic_seq = crowsetta.formats.seq.GenericSeq(annots=annots)
        generic_seq.to_file(annot_file)
        # and we now change `annot_format` as well. Both these will get passed to io.prep_spectrogram_dataset
        annot_format = 'generic-seq'

    # NOTE we set up logging here (instead of cli) so the prep log is included in the dataset
    config_logging_for_cli(
        log_dst=dataset_path,
        log_stem="prep",
        level="INFO",
        force=True
    )
    log_version(logger)

    dataset_csv_path = prep_helper.get_dataset_csv_path(dataset_path, data_dir_name, timenow)
    logger.info(
        f"Will prepare dataset as directory: {dataset_path}"
    )

    # ---- actually make the dataset -----------------------------------------------------------------------------------
    dataset_df = prep_spectrogram_dataset(
        labelset=labelset,
        data_dir=data_dir,
        annot_format=annot_format,
        annot_file=annot_file,
        audio_format=audio_format,
        spect_format=spect_format,
        spect_params=spect_params,
        spect_output_dir=dataset_path,
        audio_dask_bag_kwargs=audio_dask_bag_kwargs,
    )

    if dataset_df.empty:
        raise ValueError(
            "Calling `vak.prep.spectrogram_dataset.prep_spectrogram_dataset` "
            "with arguments passed to `vak.core.prep` "
            "returned an empty dataframe.\n"
            "Please double-check arguments to `vak.core.prep` function."
        )

    # ---- (possibly) split into train / val / test sets ---------------------------------------------
    # catch case where user specified duration for just training set, raise a helpful error instead of failing silently
    if (purpose == "train" or purpose == "learncurve") and (
        (train_dur is not None and train_dur > 0)
        and (val_dur is None or val_dur == 0)
        and (test_dur is None or val_dur == 0)
    ):
        raise ValueError(
            "duration specified for just training set, but prep function does not currently support creating a "
            "single split of a specified duration. Either remove the train_dur option from the prep section and "
            "rerun, in which case all data will be included in the training set, or specify values greater than "
            "zero for test_dur (and val_dur, if a validation set will be used)"
        )

    if all([dur is None for dur in (train_dur, val_dur, test_dur)]) or purpose in (
        "eval",
        "predict",
    ):
        # then we're not going to split
        logger.info("will not split dataset")
        do_split = False
    else:
        if val_dur is not None and train_dur is None and test_dur is None:
            raise ValueError(
                "cannot specify only val_dur, unclear how to split dataset into training and test sets"
            )
        else:
            logger.info("will split dataset")
            do_split = True

    if do_split:
        # save before splitting, jic duration args are not valid (we can't know until we make dataset)
        dataset_df.to_csv(dataset_csv_path)
        dataset_df = split.dataframe(
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
            split_name = "test"  # 'split_name' to avoid name clash with split package
        elif purpose == "predict":
            split_name = "predict"

        dataset_df = prep_helper.add_split_col(dataset_df, split=split_name)

    # ---- move prepared files into sub-directories --------------------------------------------------------------------
    prep_helper.move_files_into_split_subdirs(
        dataset_df,
        dataset_path,
        purpose
    )

    # ---- save csv file representing dataset --------------------------------------------------------------------------
    logger.info(
        f"Saving dataset csv file: {dataset_csv_path}"
    )
    dataset_df.to_csv(
        dataset_csv_path, index=False
    )  # index is False to avoid having "Unnamed: 0" column when loading

    # ---- create and save labelmap ------------------------------------------------------------------------------------
    if purpose != 'predict':
        # TODO -- add option to generate predict using existing dataset, so we can get labelmap from it
        has_unlabeled = datasets.seq.validators.has_unlabeled(dataset_csv_path, timebins_key)
        if has_unlabeled:
            map_unlabeled = True
        else:
            map_unlabeled = False
        labelmap = labels.to_map(labelset, map_unlabeled=map_unlabeled)
        logger.info(
            f"number of classes in labelmap: {len(labelmap)}",
        )
        # save labelmap in case we need it later
        with (dataset_path / "labelmap.json").open("w") as fp:
            json.dump(labelmap, fp)

    # ---- save metadata -----------------------------------------------------------------------------------------------
    # we do this before generating learncurve splits because learncurve expects metadata to exist, to get timebin_dur
    timebin_dur = prep_helper.validate_and_get_timebin_dur(dataset_df)

    metadata = Metadata(
        dataset_csv_filename=str(dataset_csv_path.name),
        timebin_dur=timebin_dur
    )
    metadata.to_json(dataset_path)

    # ---- if purpose is learncurve, additionally prep splits for that -------------------------------------------------
    if purpose == 'learncurve':
        make_learncurve_splits_from_dataset_df(
            dataset_df,
            dataset_csv_path,
            train_set_durs,
            num_replicates,
            dataset_path,
            window_size,
            labelmap,
            spect_key,
            timebins_key,
        )

    return dataset_df, dataset_path
