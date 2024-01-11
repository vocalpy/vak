"""Prepare datasets for VAE models."""
from __future__ import annotations

import logging
import pathlib
import warnings

import crowsetta

from ... import datasets
from ...common.converters import expanded_user_path, labelset_to_set
from ...common.logging import config_logging_for_cli, log_version
from ...common.timenow import get_timenow_as_str
from .. import dataset_df_helper
from .segment_vae import prep_segment_vae_dataset
from .window_vae import prep_window_vae_dataset


logger = logging.getLogger(__name__)


VAE_DATASET_TYPES = {
    "vae-segment", "vae-window"
}


def prep_vae_dataset(
    data_dir: str | pathlib.Path,
    purpose: str,
    dataset_type: str,
    output_dir: str | pathlib.Path | None = None,
    audio_format: str | None = None,
    spect_format: str | None = None,
    spect_params: dict | None = None,
    annot_format: str | None = None,
    annot_file: str | pathlib.Path | None = None,
    labelset: set | None = None,
    audio_dask_bag_kwargs: dict | None = None,
    context_s: float = 0.015,
    max_dur: float | None = None,
    target_shape: tuple[int, int] | None = None,
    train_dur: int | None = None,
    val_dur: int | None = None,
    test_dur: int | None = None,
    train_set_durs: list[float] | None = None,
    num_replicates: int | None = None,
    spect_key: str = "s",
    timebins_key: str = "t",
):
    """Prepare datasets for VAE models.

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
    dataset_type : str
        Type of VAE dataset. One of {"segment-vae", "window-vae"}.
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
    context_s : float
        Number of seconds of "context" around a segment to
        add, i.e., time before and after the onset
        and offset respectively. Default is 0.005s,
        5 milliseconds. This parameter is only used for
        Parametric UMAP and segment-VAE datasets.
    max_dur : float
        Maximum duration for segments.
        If a float value is specified,
        any segment with a duration larger than
        that value (in seconds) will be omitted
        from the dataset. Default is None.
        This parameter is only used for
        segment-VAE datasets.
    target_shape : tuple
        Of ints, (target number of frequency bins,
        target number of time bins).
        Spectrograms of units will be reshaped
        by interpolation to have the specified
        number of frequency and time bins.
        The transformation is only applied if both this
        parameter and ``max_dur`` are specified.
        Default is None.
        This parameter is only used for
        segment-VAE datasets.
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

    if dataset_type not in VAE_DATASET_TYPES:
        raise ValueError(
            f"`dataset_type` must be one of '{VAE_DATASET_TYPES}', but was: {dataset_type}"
        )
    logger.info(f"Type of VAE dataset that will be prepared : {dataset_type}")

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
                "This would cause an error because the ``prep_spectrogram_dataset`` function will attempt to "
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
                "Please specify a ``labelset`` when calling ``vak.prep.vae.prep_vae_dataset`` "
                f"with ``purpose='{purpose}'."
            )

    logger.info(f"Purpose for VAE dataset: {purpose}")
    # ---- set up directory that will contain dataset, and csv file name -----------------------------------------------
    data_dir_name = data_dir.name
    timenow = get_timenow_as_str()
    dataset_path = (
        output_dir
        / f"{data_dir_name}-vak-vae-dataset-generated-{timenow}"
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
    logger.info(f"Preparing files for '{dataset_type}' dataset")
    if dataset_type == 'vae-segment':
        dataset_df, shape = prep_segment_vae_dataset(
            data_dir,
            dataset_path,
            dataset_csv_path,
            purpose,
            audio_format,
            spect_params,
            annot_format,
            annot_file,
            labelset,
            context_s,
            max_dur,
            target_shape,
            normalize,
            train_dur,
            val_dur,
            test_dur,
            train_set_durs,
            num_replicates,
            spect_key,
            timebins_key,
        )
    elif dataset_type == 'vae-window':
        dataset_df = prep_window_vae_dataset(
            data_dir,
            dataset_path,
            dataset_csv_path,
            purpose,
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
        # only segment-vae dataset has shape -- we set to None for metadata below
        shape = None

    # ---- save csv file that captures provenance of source data -------------------------------------------------------
    logger.info(f"Saving dataset csv file: {dataset_csv_path}")
    dataset_df.to_csv(
        dataset_csv_path, index=False
    )  # index is False to avoid having "Unnamed: 0" column when loading

    # ---- save metadata -----------------------------------------------------------------------------------------------
    metadata = datasets.vae.Metadata(
        dataset_csv_filename=str(dataset_csv_path.name),
        dataset_type=dataset_type,
        audio_format=audio_format,
        shape=shape,
    )
    metadata.to_json(dataset_path)

    return dataset_df, dataset_path
