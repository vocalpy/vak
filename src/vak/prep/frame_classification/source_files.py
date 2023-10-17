import logging
import pathlib

import pandas as pd

from ...common.converters import expanded_user_path, labelset_to_set
from .. import constants
from ..audio_dataset import prep_audio_dataset
from ..spectrogram_dataset.prep import prep_spectrogram_dataset

logger = logging.getLogger(__name__)


def get_or_make_source_files(
    data_dir: str | pathlib.Path,
    input_type: str,
    audio_format: str | None = None,
    spect_format: str | None = None,
    spect_params: dict | None = None,
    spect_output_dir: str | pathlib.Path | None = None,
    annot_format: str | None = None,
    annot_file: str | pathlib.Path | None = None,
    labelset: set | None = None,
    audio_dask_bag_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Get source files for a dataset, or make them.

    Gets either audio or spectrogram files from ``data dir``,
    possibly paired with annotation files.

    If ``input_type`` is ``'audio'``, then this function will look
    for files with the extension for ``audio_format`` in ``data_dir``.
    If ``input_type`` is ``'spectrogram'``, and ``spect_format`` is specified,
    then this function will look for files with the extension for that format
    in ``data_dir``. If ``input_type`` is spectrogram,
    and ``audio_format`` is specified,
    this function will look for audio files with that extension
    and then generate spectrograms for them using ``spect_params``.
    If an ``annot_format`` is specified, this function will additionally
    look for annotation files for the audio or spectrogram files.
    If all annotations are in a single file, this can be specified
    with the ``annot_file`` parameter, and that will be used instead
    of looking for other annotation files.

    Parameters
    ----------
    data_dir : str, Path
        Path to directory with files from which to make dataset.
    input_type : str
        The type of input to the neural network model.
        One of {'audio', 'spect'}.
    audio_format : str
        Format of audio files. One of {'wav', 'cbin'}.
        Default is ``None``, but either ``audio_format`` or ``spect_format``
        must be specified.
    spect_format : str
        Format of files containing spectrograms as 2-d matrices. One of {'mat', 'npz'}.
        Default is None, but either audio_format or spect_format must be specified.
    spect_params : dict, vak.config.SpectParams
        Parameters for creating spectrograms. Default is ``None``.
    spect_output_dir : str
        Path to location where spectrogram files should be saved.
        Default is None. If ``input_type`` is ``'spect'``,
        then ``spect_output_dir`` defaults to ``data_dir``.
    annot_format : str
        Format of annotations. Any format that can be used with the
        :module:`crowsetta` library is valid. Default is ``None``.
    annot_file : str
        Path to a single annotation file. Default is ``None``.
        Used when a single file contains annotates multiple audio
        or spectrogram files.
    audio_dask_bag_kwargs : dict
        Keyword arguments used when calling :func:`dask.bag.from_sequence`
        inside :func:`vak.io.audio`, where it is used to parallelize
        the conversion of audio files into spectrograms.
        Option should be specified in config.toml file as an inline table,
        e.g., ``audio_dask_bag_kwargs = { npartitions = 20 }``.
        Allows for finer-grained control
        when needed to process files of different sizes.
    labelset : str, list, set
        Set of unique labels for vocalizations. Strings or integers.
        Default is ``None``. If not ``None``, then files will be skipped
        where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using
        :func:`vak.converters.labelset_to_set`.
        See help for that function for details on how to specify ``labelset``.

    Returns
    -------
    source_files_df : pandas.DataFrame
        Source files that will become the dataset,
        represented as a pandas.DataFrame.
        Each row corresponds to one sample in the dataset,
        either an audio file or spectrogram file,
        possibly paired with annotations.
    """
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

    if annot_file is not None:
        annot_file = expanded_user_path(annot_file)
        if not annot_file.exists():
            raise FileNotFoundError(
                f"Path specified for ``annot_file`` not found: {annot_file}"
            )

    if input_type == "spect":
        source_files_df = prep_spectrogram_dataset(
            data_dir,
            annot_format,
            labelset,
            annot_file,
            audio_format,
            spect_format,
            spect_params,
            spect_output_dir,
            audio_dask_bag_kwargs,
        )
        if source_files_df.empty:
            raise ValueError(
                "Calling `vak.prep.spectrogram_dataset.prep_spectrogram_dataset` "
                "with arguments passed to `vak.prep.prep_frame_classification_dataset` "
                "returned an empty dataframe.\n"
                "Please double-check arguments to `prep_frame_classification_dataset` function."
            )

    elif input_type == "audio":
        source_files_df = prep_audio_dataset(
            audio_format,
            data_dir,
            annot_format,
            labelset,
        )
        if source_files_df.empty:
            raise ValueError(
                "Calling `vak.prep.audio_dataset.prep_audio_dataset` "
                "with arguments passed to `vak.prep.prep_frame_classification_dataset` "
                "returned an empty dataframe.\n"
                "Please double-check arguments to `prep_frame_classification_dataset` function."
            )

    return source_files_df
