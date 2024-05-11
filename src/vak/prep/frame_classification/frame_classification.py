"""Function that prepares datasets for neural network models
that perform the frame classification task."""

from __future__ import annotations

import json
import logging
import pathlib
import warnings

import crowsetta.formats.seq
import pandas as pd

from ... import datapipes
from ...common import labels
from ...common.converters import expanded_user_path, labelset_to_set
from ...common.logging import config_logging_for_cli, log_version
from ...common.timenow import get_timenow_as_str
from .. import dataset_df_helper, sequence_dataset
from . import validators
from .assign_samples_to_splits import assign_samples_to_splits
from .learncurve import make_subsets_from_dataset_df
from .make_splits import make_splits
from .source_files import get_or_make_source_files

logger = logging.getLogger(__name__)


def prep_frame_classification_dataset(
    data_dir: str | pathlib.Path,
    input_type: str,
    purpose: str,
    output_dir: str | pathlib.Path | None = None,
    audio_format: str | None = None,
    spect_format: str | None = None,
    spect_params: dict | None = None,
    annot_format: str | None = None,
    annot_file: str | pathlib.Path | None = None,
    labelset: set | None = None,
    audio_dask_bag_kwargs: dict | None = None,
    train_dur: float | None = None,
    val_dur: float | None = None,
    test_dur: float | None = None,
    train_set_durs: list[float] | None = None,
    num_replicates: int | None = None,
    spect_key: str = "s",
    timebins_key: str = "t",
    freqbins_key: str = "f",
):
    """Prepare datasets for neural network models
    that perform the frame classification task.

    For general information on dataset preparation,
    see the docstring for :func:`vak.prep.prep`.

    Parameters
    ----------
    data_dir : str, Path
        Path to directory with files from which to make dataset.
    input_type : str
        The type of input to the neural network model.
        One of {'audio', 'spect'}.
    purpose : str
        Purpose of the dataset.
        One of {'train', 'eval', 'predict', 'learncurve'}.
        These correspond to commands of the vak command-line interface.
    input_type : str
        The type of input to the neural network model.
        One of {'audio', 'spect'}.
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
    spect_key : str
        Key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        Key for accessing vector of time bins in files. Default is 't'.
    freqbins_key : str
        Key for accessing vector of frequency bins in files. Default is 'f'.

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

    if input_type not in constants.INPUT_TYPES:
        raise ValueError(
            f"``input_type`` must be one of: {constants.INPUT_TYPES}\n"
            f"Value for ``input_type`` was: {input_type}"
        )

    if input_type == "audio" and spect_format is not None:
        raise ValueError(
            f"Input type was 'audio' but a ``spect_format`` was specified: '{spect_format}'. "
            f"Please specify ``audio_format`` only."
        )

    if input_type == "audio" and audio_format is None:
        raise ValueError(
            "Input type was 'audio' but no ``audio_format`` was specified. "
        )

    if audio_format is None and spect_format is None:
        raise ValueError(
            "Must specify either ``audio_format`` or ``spect_format``"
        )

    if audio_format and spect_format:
        raise ValueError(
            "Cannot specify both ``audio_format`` and ``spect_format``, "
            "unclear whether to create spectrograms from audio files or "
            "use already-generated spectrograms from array files"
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
        / f"{data_dir_name}-vak-frame-classification-dataset-generated-{timenow}"
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

    # ---- get or make source files: either audio or spectrogram, possible paired with annotation files ----------------
    source_files_df: pd.DataFrame = get_or_make_source_files(
        data_dir,
        input_type,
        audio_format,
        spect_format,
        spect_params,
        dataset_path,
        annot_format,
        annot_file,
        labelset,
        audio_dask_bag_kwargs,
    )

    # save before (possibly) splitting, just in case duration args are not valid
    # (we can't know until we make dataset)
    source_files_df.to_csv(dataset_csv_path)

    # ---- assign samples to splits; adds a 'split' column to dataset_df, calling `vak.prep.split` if needed -----------
    # once we assign a split, we consider this the ``dataset_df``
    dataset_df: pd.DataFrame = assign_samples_to_splits(
        purpose,
        source_files_df,
        dataset_path,
        train_dur,
        val_dur,
        test_dur,
        labelset,
    )

    # ---- create and save labelmap ------------------------------------------------------------------------------------
    # we do this before creating array files since we need to load the labelmap to make frame label vectors
    if purpose != "predict":
        # TODO: add option to generate predict using existing dataset, so we can get labelmap from it
        map_unlabeled_segments = sequence_dataset.has_unlabeled_segments(
            dataset_df
        )
        labelmap = labels.to_map(
            labelset, map_background=map_unlabeled_segments
        )
        logger.info(
            f"Number of classes in labelmap: {len(labelmap)}",
        )
        # save labelmap in case we need it later
        with (dataset_path / "labelmap.json").open("w") as fp:
            json.dump(labelmap, fp)
    else:
        labelmap = None

    # ---- actually move/copy/create files into directories representing splits ----------------------------------------
    # now we're *remaking* the dataset_df (actually adding additional rows with the splits)
    dataset_df: pd.DataFrame = make_splits(
        dataset_df,
        dataset_path,
        input_type,
        purpose,
        labelmap,
        audio_format,
        spect_key,
        timebins_key,
        freqbins_key,
    )

    # ---- if purpose is learncurve, additionally prep training data subsets for the learning curve --------------------
    if purpose == "learncurve":
        dataset_df: pd.DataFrame = make_subsets_from_dataset_df(
            dataset_df,
            input_type,
            train_set_durs,
            num_replicates,
            dataset_path,
            labelmap,
        )

    # ---- save csv file that captures provenance of source data -------------------------------------------------------
    logger.info(f"Saving dataset csv file: {dataset_csv_path}")
    dataset_df.to_csv(
        dataset_csv_path, index=False
    )  # index is False to avoid having "Unnamed: 0" column when loading

    # ---- save metadata -----------------------------------------------------------------------------------------------
    frame_dur = validators.validate_and_get_frame_dur(dataset_df, input_type)

    if input_type == "spect" and spect_format != "npz":
        # then change to npz since we canonicalize data so it's always npz arrays
        # We need this to be correct for other functions, e.g. predict when it loads spectrogram files
        spect_format = "npz"

    metadata = datapipes.frame_classification.Metadata(
        dataset_csv_filename=str(dataset_csv_path.name),
        frame_dur=frame_dur,
        input_type=input_type,
        audio_format=audio_format,
        spect_format=spect_format,
    )
    metadata.to_json(dataset_path)

    return dataset_df, dataset_path
