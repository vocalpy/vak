import logging
import pathlib

import pandas as pd

from ..audio_dataset import prep_audio_dataset
from ..spectrogram_dataset.prep import prep_spectrogram_dataset

logger = logging.getLogger(__name__)


def get_or_make_source_files(
        data_dir: str | pathlib.Path,
        input_type: str,
        annot_format: str,
        annot_file: str | pathlib.Path,
        audio_format: str,
        spect_format: str,
        spect_params: dict,
        spect_output_dir: str | pathlib.Path,
        audio_dask_bag_kwargs: dict,
        labelset: set | None = None,
) -> pd.DataFrame:
    """Get source files for dataset, or make them.

    Gets either audio or spectrogram files from ``data dir``,
    possibly paired with annotation files.

    Parameters
    ----------
    input_type : str
        The type of input to the neural network model.
        One of {'audio', 'spect'}.
    data_dir : str, Path
        Path to directory with files from which to make dataset.
    annot_format : str
        Format of annotations. Any format that can be used with the
        :module:`crowsetta` library is valid. Default is ``None``.
    annot_file : str
        Path to a single annotation file. Default is ``None``.
        Used when a single file contains annotates multiple audio
        or spectrogram files.
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
        Default is None, in which case it defaults to ``data_dir``.
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
    dataset_df : pandas.DataFrame
        Source files that will become the dataset,
        represented as a pandas.DataFrame.
        Each row corresponds to one sample in the dataset,
        either an audio file or spectrogram file,
        possibly paired with annotations.
    """
    if input_type == "spect":
        dataset_df = prep_spectrogram_dataset(
            labelset,
            data_dir,
            annot_format,
            annot_file,
            audio_format,
            spect_format,
            spect_params,
            spect_output_dir,
            audio_dask_bag_kwargs,
        )
        if dataset_df.empty:
            raise ValueError(
                "Calling `vak.prep.spectrogram_dataset.prep_spectrogram_dataset` "
                "with arguments passed to `vak..prep.prep_frame_classification_dataset` "
                "returned an empty dataframe.\n"
                "Please double-check arguments to `prep_frame_classification_dataset` function."
            )

    elif input_type == "audio":
        dataset_df = prep_audio_dataset(
            audio_format,
            data_dir,
            annot_format,
            labelset,
        )
        if dataset_df.empty:
            raise ValueError(
                "Calling `vak.prep.audio_dataset.prep_audio_dataset` "
                "with arguments passed to `vak.prep.prep_frame_classification_dataset` "
                "returned an empty dataframe.\n"
                "Please double-check arguments to `prep_frame_classification_dataset` function."
            )

    return dataset_df
