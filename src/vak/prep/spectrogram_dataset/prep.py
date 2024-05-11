from __future__ import annotations

import logging
import pathlib

import attrs
import crowsetta
import pandas as pd

from ...common import annotation, constants
from ...common.converters import expanded_user_path, labelset_to_set
from ...config.spect_params import SpectParamsConfig
from . import audio_helper, spect_helper

logger = logging.getLogger(__name__)


def prep_spectrogram_dataset(
    data_dir: str | pathlib.Path,
    annot_format: str | None = None,
    labelset: set | None = None,
    annot_file: str | pathlib.Path | None = None,
    audio_format: str | None = None,
    spect_format: str | None = None,
    spect_params: dict | None = None,
    spect_output_dir: str | pathlib.Path | None = None,
    audio_dask_bag_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Make a dataset of spectrograms,
    optionally paired with annotations.

    Prepares dataset of vocalizations from a directory of audio or spectrogram files,
    and (optionally) annotation for those files. The dataset is returned as a pandas DataFrame.

    Datasets are used to train neural networks, predicting annotations for
    the dataset itself using a trained neural network, etc.

    If dataset is created from audio files, then array files containing spectrograms
    will be generated from the audio files and saved in ``spect_output_dir``
    with the extension ``.spect.npz``. The ``spect_output_dir`` defaults to ``data_dir``
    if is not specified.

    Parameters
    ----------
    data_dir : str
        path to directory with audio or spectrogram files from which to make dataset
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid. Default is None.
    labelset : str, list, set
        of str or int, set of unique labels for vocalizations. Default is None.
        If not None, then files will be skipped where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using ``vak.converters.labelset_to_set``.
        See help for that function for details on how to specify labelset.
    load_spects : bool
        if True, load spectrograms. If False, return a InferDatapipe without spectograms loaded.
        Default is True. Set to False when you want to create a InferDatapipe for use
        later, but don't want to load all the spectrograms into memory yet.
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
    spect_format : str
        format of array files containing spectrograms as 2-d matrices.
        One of {'mat', 'npz'}.
    annot_file : str
        Path to a single annotation file. Default is None.
        Used when a single file contains annotations for multiple audio files.
    spect_params : dict, vak.config.spect.SpectParamsConfig.
        Parameters for creating spectrograms.
        Default is None (implying that spectrograms are already made).
    spect_output_dir : str
        Path to location where spectrogram files should be saved.
        Default is None, in which case it defaults to ``data_dir``.
    audio_dask_bag_kwargs : dict
        Keyword arguments used when calling ``dask.bag.from_sequence``
        inside ``vak.io.audio``, where it is used to parallelize
        the conversion of audio files into spectrograms.
        Option should be specified in config.toml file as an inline table,
        e.g., ``audio_dask_bag_kwargs = { npartitions = 20 }``.
        Allows for finer-grained control
        when needed to process files of different sizes.

    Returns
    -------
    source_files_df : pandas.DataFrame
        A set of source files that will be used to prepare a
        data set for use with neural network models,
        represented as a :class:`pandas.DataFrame`.
        Will contain paths to spectrogram files,
        possibly paired with annotation files,
        as well as the original audio files if the
        spectrograms were generated from audio by
        :func:`vak.prep.audio_helper.make_spectrogram_files_from_audio_files`.
        The columns of the dataframe are specified by
        :const:`vak.prep.spectrogram_dataset.spect_helper.DF_COLUMNS`.
    """
    # ---- pre-conditions ----------------------------------------------------------------------------------------------
    if labelset is not None:
        labelset = labelset_to_set(labelset)

    if audio_format is None and spect_format is None:
        raise ValueError("Must specify either audio_format or spect_format")

    if audio_format and spect_format:
        raise ValueError(
            "Cannot specify both audio_format and spect_format, "
            "unclear whether to create spectrograms from audio files or "
            "use already-generated spectrograms from array files"
        )

    data_dir = expanded_user_path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir not found: {data_dir}")

    if spect_output_dir:
        spect_output_dir = expanded_user_path(spect_output_dir)
        if not spect_output_dir.is_dir():
            raise NotADirectoryError(
                f"spect_output_dir not found: {spect_output_dir}"
            )
    else:
        spect_output_dir = data_dir

    if annot_format is not None:
        if annot_file is None:
            annot_files = annotation.files_from_dir(
                annot_dir=data_dir, annot_format=annot_format
            )
            scribe = crowsetta.Transcriber(format=annot_format)
            annot_list = [
                scribe.from_file(annot_file).to_annot()
                for annot_file in annot_files
            ]
        else:
            scribe = crowsetta.Transcriber(format=annot_format)
            annot_list = scribe.from_file(annot_file).to_annot()
        if isinstance(annot_list, crowsetta.Annotation):
            # if e.g. only one annotated audio file in directory, wrap in a list to make iterable
            # fixes https://github.com/NickleDave/vak/issues/467
            annot_list = [annot_list]
    else:  # if annot_format not specified
        annot_list = None

    # ------ if making dataset from audio files, need to make into array files first! ----------------------------------
    if audio_format:
        logger.info(
            f"making array files containing spectrograms from audio files in: {data_dir}",
        )
        audio_files = audio_helper.files_from_dir(data_dir, audio_format)

        spect_files = audio_helper.make_spectrogram_files_from_audio_files(
            audio_format=audio_format,
            spect_params=spect_params,
            output_dir=spect_output_dir,
            audio_files=audio_files,
            annot_list=annot_list,
            annot_format=annot_format,
            labelset=labelset,
            dask_bag_kwargs=audio_dask_bag_kwargs,
        )
        spect_format = "npz"
        spect_ext = constants.SPECT_NPZ_EXTENSION
    else:  # if audio format is None
        spect_files = None
        # make sure we use the vak extension for spectrogram files
        spect_ext = constants.SPECT_FORMAT_EXT_MAP[spect_format]

    make_dataframe_kwargs = {
        "spect_format": spect_format,
        "labelset": labelset,
        "annot_list": annot_list,
        "annot_format": annot_format,
        "spect_ext": spect_ext,
    }

    if (
        spect_files
    ):  # because we just made them, and put them in spect_output_dir
        make_dataframe_kwargs["spect_files"] = spect_files
        logger.info(
            f"creating dataset from spectrogram files in: {spect_output_dir}",
        )
    else:
        make_dataframe_kwargs["spect_dir"] = data_dir
        logger.info(
            f"creating dataset from spectrogram files in: {data_dir}",
        )

    if spect_params:  # get relevant keys for accessing arrays from array files
        if isinstance(spect_params, SpectParamsConfig):
            spect_params = attrs.asdict(spect_params)
        for key in [
            "freqbins_key",
            "timebins_key",
            "spect_key",
            "audio_path_key",
        ]:
            make_dataframe_kwargs[key] = spect_params[key]

    source_files_df = spect_helper.make_dataframe_of_spect_files(
        **make_dataframe_kwargs
    )
    return source_files_df
