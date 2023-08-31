from __future__ import annotations

import logging
import pathlib

from . import constants
from .frame_classification import prep_frame_classification_dataset
from .parametric_umap import prep_parametric_umap_dataset

logger = logging.getLogger(__name__)


def prep(
    data_dir: str | pathlib.Path,
    purpose: str,
    dataset_type: str,
    input_type: str,
    output_dir: str | pathlib.Path | None = None,
    audio_format: str | None = None,
    spect_format: str | None = None,
    spect_params: dict | None = None,
    annot_format: str | None = None,
    annot_file: str | pathlib.Path | None = None,
    labelset: set | None = None,
    audio_dask_bag_kwargs: dict | None = None,
    train_dur: int | None = None,
    val_dur: int | None = None,
    test_dur: int | None = None,
    train_set_durs: list[float] | None = None,
    num_replicates: int | None = None,
    spect_key: str = "s",
    timebins_key: str = "t",
    context_s: float = 0.015,
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
    dataset_type : str
        String name of the type of dataset, e.g.,
        'frame_classification'. Dataset types are
        defined by machine learning tasks, e.g.,
        a 'frame_classification' dataset would be used
        a :class:`vak.models.FrameClassificationModel` model.
        Valid dataset types are defined as
        :const:`vak.prep.prep.DATASET_TYPES`.
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
        :mod:`crowsetta` library is valid. Default is ``None``.
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
    if purpose not in constants.VALID_PURPOSES:
        raise ValueError(
            f"purpose must be one of: {constants.VALID_PURPOSES}\n"
            f"Value for purpose was: {purpose}"
        )

    if dataset_type not in constants.DATASET_TYPES:
        raise ValueError(
            f"``dataset_type`` must be one of: {constants.DATASET_TYPES}\n"
            f"Value for ``dataset_type`` was: {dataset_type}"
        )

    if input_type not in constants.INPUT_TYPES:
        raise ValueError(
            f"``input_type`` must be one of: {constants.INPUT_TYPES}\n"
            f"Value for ``input_type`` was: {input_type}"
        )

    if input_type == "audio" and spect_format is not None:
        raise ValueError(
            f"``input_type`` was set to 'audio' but a ``spect_format`` was specified: '{spect_format}'.\n"
            f"Please only provide a ``spect_format`` argument when the input type to the neural network "
            f"model is spectrograms."
        )

    if audio_format is None and spect_format is None:
        raise ValueError("Must specify either audio_format or spect_format")

    if audio_format and spect_format:
        raise ValueError(
            "Cannot specify both audio_format and spect_format, "
            "unclear whether to compute spectrograms from audio files or "
            "use pre-computed spectrograms from existing array files."
        )

    # we have to use an if-else here since args may vary across dataset prep functions
    # but we still define DATASET_TYPE_FUNC_MAP in vak.prep.constants
    # so that the mapping is made explicit in the code
    if dataset_type == "frame classification":
        dataset_df, dataset_path = prep_frame_classification_dataset(
            data_dir,
            input_type,
            purpose,
            output_dir,
            audio_format,
            spect_format,
            spect_params,
            annot_format,
            annot_file,
            labelset,
            audio_dask_bag_kwargs,
            train_dur,
            val_dur,
            test_dur,
            train_set_durs,
            num_replicates,
            spect_key,
            timebins_key,
        )
        return dataset_df, dataset_path
    elif dataset_type == "parametric umap":
        dataset_df, dataset_path = prep_parametric_umap_dataset(
            data_dir,
            purpose,
            output_dir,
            audio_format,
            spect_params,
            annot_format,
            annot_file,
            labelset,
            context_s,
            train_dur,
            val_dur,
            test_dur,
            train_set_durs,
            num_replicates,
            spect_key=spect_key,
            timebins_key=timebins_key,
        )
        return dataset_df, dataset_path
    else:
        # this is in case a dataset type is written wrong
        # in the if-else statements above, we want to error loudly
        raise ValueError(f"Unrecognized dataset type: {dataset_type}")
