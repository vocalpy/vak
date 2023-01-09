from datetime import datetime
import logging
from pathlib import Path
import warnings

from .. import split
from ..converters import expanded_user_path, labelset_to_set
from ..io import dataframe


logger = logging.getLogger(__name__)


VALID_PURPOSES = frozenset(
    [
        "eval",
        "learncurve",
        "predict",
        "train",
    ]
)


def prep(
    data_dir,
    purpose,
    output_dir=None,
    audio_format=None,
    spect_format=None,
    spect_params=None,
    spect_output_dir=None,
    annot_format=None,
    annot_file=None,
    labelset=None,
    audio_dask_bag_kwargs=None,
    train_dur=None,
    val_dur=None,
    test_dur=None,
):
    """Prepare datasets of vocalizations for use with neural network models.

    High-level function that prepares datasets to be used by other
    high-level functions like vak.train, vak.predict, and vak.learncurve

    Saves a .csv file representing the dataset generated from data_dir.

    Datasets are used to train neural networks that segment audio files into
    vocalizations, and then predict labels for those segments.
    The function also prepares datasets so neural networks can predict the
    segmentation and annotation of vocalizations in them.
    It can also split a dataset into training, validation, and test sets,
    e.g. for benchmarking different neural network architectures.

    If the 'purpose' is set to 'train' or 'learncurve', and/or
    the duration of either the training or test set is provided,
    then the function attempts to split the dataset into training and test sets.
    A duration can also be specified for a validation set
    (used to measure performance during training).
    In these cases, the 'split' column in the .csv
    identifies which files (rows) belong to the training, test, and
    validation sets created from that Dataset.

    If the 'purpose' is set to 'predict' or 'eval',
    or no durations for any of the training sets are specified,
    then the function assumes all the vocalizations constitute a single
    dataset, and for all rows the 'split' columns for that dataset
    will be 'predict' or 'test' (respectively).

    Parameters
    ----------
    data_dir : str, Path
        path to directory with files from which to make dataset
    purpose : str
        one of {'train', 'predict', 'learncurve'}
    output_dir : str
        Path to location where data sets should be saved.
        Default is None, in which case data sets to `data_dir`.
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
        Default is None, but either audio_format or spect_format must be specified.
    spect_format : str
        format of files containing spectrograms as 2-d matrices. One of {'mat', 'npz'}.
        Default is None, but either audio_format or spect_format must be specified.
    spect_params : dict, vak.config.SpectParams
        parameters for creating spectrograms. Default is None.
    spect_output_dir : str
        path to location where spectrogram files should be saved.
        Default is None, in which case it defaults to ``data_dir``.
        A new directory will be created in ``spect_output_dir`` with
        the name 'spectrograms_generated_{time stamp}'.
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid. Default is None.
    annot_file : str
        Path to a single annotation file. Default is None.
        Used when a single file contains annotations for multiple audio files.
    labelset : str, list, set
        of str or int, set of unique labels for vocalizations. Default is None.
        If not None, then files will be skipped where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using ``vak.converters.labelset_to_set``.
        See help for that function for details on how to specify labelset.
    audio_dask_bag_kwargs : dict
        Keyword arguments used when calling ``dask.bag.from_sequence``
        inside ``vak.io.audio``, where it is used to parallelize
        the conversion of audio files into spectrograms.
        Option should be specified in config.toml file as an inline table,
        e.g., ``audio_dask_bag_kwargs = { npartitions = 20 }``.
        Allows for finer-grained control
        when needed to process files of different sizes.
    train_dur : float
        total duration of training set, in seconds. When creating a learning curve,
        training subsets of shorter duration will be drawn from this set. Default is None.
    val_dur : float
        total duration of validation set, in seconds. Default is None.
    test_dur : float
        total duration of test set, in seconds. Default is None.

    Returns
    -------
    vak_df : pandas.DataFrame
        that represents a dataset of vocalizations
    csv_path : Path
        to csv saved from vak_df
    """
    # pre-conditions ---------------------------------------------------------------------------------------------------
    if purpose not in VALID_PURPOSES:
        raise ValueError(
            f"purpose must be one of: {VALID_PURPOSES}\nValue for purpose was: {purpose}"
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

    if spect_output_dir:
        spect_output_dir = expanded_user_path(spect_output_dir)
        if not spect_output_dir.is_dir():
            raise NotADirectoryError(f"spect_output_dir not found: {spect_output_dir}")

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
                "This would cause an error because the dataframe.from_files section will attempt to "
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
    # ---- figure out file name ----------------------------------------------------------------------------------------
    data_dir_name = data_dir.name
    timenow = datetime.now().strftime("%y%m%d_%H%M%S")
    csv_fname_stem = f"{data_dir_name}_prep_{timenow}"
    csv_path = output_dir.joinpath(f"{csv_fname_stem}.csv")

    # ---- figure out if we're going to split into train / val / test sets ---------------------------------------------
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

    # ---- actually make the dataset -----------------------------------------------------------------------------------
    vak_df = dataframe.from_files(
        labelset=labelset,
        data_dir=data_dir,
        annot_format=annot_format,
        annot_file=annot_file,
        audio_format=audio_format,
        spect_format=spect_format,
        spect_params=spect_params,
        spect_output_dir=spect_output_dir,
        audio_dask_bag_kwargs=audio_dask_bag_kwargs,
    )

    if vak_df.empty:
        raise ValueError(
            "Calling `vak.io.dataframe.from_files` with arguments passed to `vak.core.prep` "
            "returned an empty dataframe.\n"
            "Please double-check arguments to `vak.core.prep` function."
        )

    if do_split:
        # save before splitting, jic duration args are not valid (we can't know until we make dataset)
        vak_df.to_csv(csv_path)
        vak_df = split.dataframe(
            vak_df,
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

        vak_df = dataframe.add_split_col(vak_df, split=split_name)

    logger.info(
        f"saving dataset as a .csv file: {csv_path}"
    )
    vak_df.to_csv(
        csv_path, index=False
    )  # index is False to avoid having "Unnamed: 0" column when loading

    return vak_df, csv_path
