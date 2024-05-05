from __future__ import annotations

import logging
import pathlib

import crowsetta
import dask.bag as db
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from ..common import annotation, constants
from ..common.converters import expanded_user_path, labelset_to_set
from ..common.typing import PathLike
from .spectrogram_dataset.audio_helper import files_from_dir

logger = logging.getLogger(__name__)


# constant, used for names of columns in DataFrame below
DF_COLUMNS = [
    "audio_path",
    "annot_path",
    "annot_format",
    "samplerate",
    "sample_dur",
    "duration",
]


def prep_audio_dataset(
    data_dir: PathLike,
    audio_format: str,
    annot_format: str | None = None,
    annot_file: str | pathlib.Path | None = None,
    labelset: set | None = None,
) -> pd.DataFrame:
    """Gets a set of audio files from a directory,
    optionally paired with an annotation file or files,
    and return a Pandas DataFrame that represents the set
    of files.

    Finds all files with ``audio_format`` in ``data_dir``,
    then finds any annotations with ``annot_format`` if specified,
    and additionally filter the audio and annotation files
    by ``labelset`` if specified.
    Then creates the dataframe with columns specified by
    ``vak.prep.audio_dataset.DF_COLUMNS``:
    ``"audio_path"``, ``"annot_path"``, ``"annot_format"``, ``"samplerate"``,
    ``"sample_dur",`` and ``"duration"``.

    Parameters
    ----------
    data_dir : str, pathlib.Path
        Path to directory containing audio files that should be used in dataset.
    audio_format : str
        A :class:`string` representing the format of audio files.
        One of :constant:`vak.common.constants.VALID_AUDIO_FORMATS`.
    annot_format : str
        Name of annotation format. Added as a column to the DataFrame if specified.
        Used by other functions that open annotation files via their paths from the DataFrame.
        Should be a format that the :mod:`crowsetta` library recognizes.
        Default is None.
    annot_file : str
        Path to a single annotation file. Default is None.
        Used when a single file contains annotations for multiple audio files.
    labelset : str, list, set
        Iterable of str or int, set of unique labels for annotations. Default is None.
        If not None, then files will be skipped where the associated annotation
        contains labels *not* found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using
        :func:`vak.common.converters.labelset_to_set`.
        See docstring of that function for details on how to specify ``labelset``.

    Returns
    -------
    source_files_df : pandas.Dataframe
        A set of source files that will be used to prepare a
        data set for use with neural network models,
        represented as a :class:`pandas.DataFrame`.
        Will contain paths to audio files,
        possibly paired with annotation files.
        The columns of the dataframe are specified by
        :const:`vak.prep.audio_dataset.DF_COLUMNS`.
    """
    # pre-conditions ---------------------------------------------------------------------------------------------------
    if audio_format not in constants.VALID_AUDIO_FORMATS:
        raise ValueError(
            f"audio format must be one of '{constants.VALID_AUDIO_FORMATS}'; "
            f"format '{audio_format}' not recognized."
        )

    if labelset is not None:
        labelset = labelset_to_set(labelset)

    data_dir = expanded_user_path(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise NotADirectoryError(
            f"`data_dir` not found, or not recognized as a directory:\n{data_dir}"
        )

    audio_files = files_from_dir(data_dir, audio_format)

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

    if annot_list:
        audio_annot_map = annotation.map_annotated_to_annot(
            audio_files, annot_list, annot_format
        )
    else:
        # no annotation, so map spectrogram files to None
        audio_annot_map = dict(
            (audio_path, None) for audio_path in audio_files
        )

    # use mapping (if generated/supplied) with labelset, if supplied, to filter
    if labelset:  # then remove annotations with labels not in labelset
        for audio_file, annot in list(audio_annot_map.items()):
            # loop in a verbose way (i.e. not a comprehension)
            # so we can give user warning when we skip files
            annot_labelset = set(annot.seq.labels)
            # below, set(labels_mapping) is a set of that dict's keys
            if not annot_labelset.issubset(set(labelset)):
                # because there's some label in labels that's not in labelset
                audio_annot_map.pop(audio_file)
                extra_labels = annot_labelset - labelset
                logger.info(
                    f"Found labels, {extra_labels}, in {pathlib.Path(audio_file).name}, "
                    "that are not in labels_mapping. Skipping file.",
                )

    # ---- actually make the dataframe ---------------------------------------------------------------------------------
    # this is defined here so all other arguments to 'to_dataframe' are in scope
    def _to_record(audio_annot_tuple):
        """helper function that enables parallelized creation of "records",
        i.e. rows for dataframe, from .
        Accepts a two-element tuple containing (1) a dictionary that represents a spectrogram
        and (2) annotation for that file"""
        audio_path, annot = audio_annot_tuple
        dat, samplerate = constants.AUDIO_FORMAT_FUNC_MAP[audio_format](
            audio_path
        )
        sample_dur = 1.0 / samplerate
        audio_dur = dat.shape[-1] * sample_dur

        if annot is not None:
            annot_path = annot.annot_path
        else:
            annot_path = np.nan

        def abspath(a_path):
            if isinstance(a_path, str) or isinstance(a_path, pathlib.Path):
                return str(pathlib.Path(a_path).absolute())
            elif np.isnan(a_path):
                return a_path

        record = tuple(
            [
                abspath(audio_path),
                abspath(annot_path),
                (
                    annot_format
                    if annot_format
                    else constants.NO_ANNOTATION_FORMAT
                ),
                samplerate,
                sample_dur,
                audio_dur,
            ]
        )
        return record

    audio_path_annot_tuples = db.from_sequence(audio_annot_map.items())
    logger.info(
        "creating pandas.DataFrame representing dataset from audio files",
    )
    with ProgressBar():
        records = list(audio_path_annot_tuples.map(_to_record))

    return pd.DataFrame.from_records(data=records, columns=DF_COLUMNS)
