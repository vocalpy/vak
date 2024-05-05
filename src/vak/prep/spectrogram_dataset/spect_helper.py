"""Function that converts a set of array files (.npz, .mat) containing spectrograms
into a pandas DataFrame that represents a dataset used by ``vak``.

The columns of the dataframe are specified by
 :const:`vak.prep.spectrogram_dataset.spect_helper.DF_COLUMNS`.
"""

from __future__ import annotations

import logging
import pathlib

import dask.bag as db
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from ...common import constants, files
from ...common.annotation import map_annotated_to_annot
from ...common.converters import labelset_to_set

logger = logging.getLogger(__name__)


# constant, used for names of columns in DataFrame below
DF_COLUMNS = [
    "audio_path",
    "spect_path",
    "annot_path",
    "annot_format",
    "duration",
    "timebin_dur",
]


def make_dataframe_of_spect_files(
    spect_format: str,
    spect_dir: str | pathlib.Path | None = None,
    spect_files: list | None = None,
    spect_ext: str | None = None,
    annot_list: list | None = None,
    annot_format: str | None = None,
    labelset: set | None = None,
    n_decimals_trunc: int = 5,
    freqbins_key: str = "f",
    timebins_key: str = "t",
    spect_key: str = "s",
    audio_path_key: str = "audio_path",
) -> pd.DataFrame:
    """Get a set of spectrogram files from a directory,
    optionally paired with an annotation file or files,
    and returns a Pandas DataFrame that represents all the files.

    Spectrogram files are array in npz files created by numpy
    or in mat files created by Matlab.

    Parameters
    ----------
    spect_format : str
        Format of files containing spectrograms. One of {'mat', 'npz'}
    spect_dir : str
        Path to directory of files containing spectrograms as arrays.
        Default is None.
    spect_files : list
        List of paths to array files. Default is None.
    annot_list : list
        List of annotations for array files. Default is None
    annot_format : str
        Name of annotation format. Added as a column to the DataFrame if specified.
        Used by other functions that open annotation files via their paths from the DataFrame.
        Should be a format that the crowsetta library recognizes.
        Default is None.
    labelset : str, list, set
        Set of unique labels for vocalizations, of str or int. Default is None.
        If not None, then files will be skipped where the associated annotation
        contains labels not found in ``labelset``.
        ``labelset`` is converted to a Python ``set`` using
        :func:`vak.common.converters.labelset_to_set`.
        See help for that function for details on how to specify labelset.
    n_decimals_trunc : int
        number of decimal places to keep when truncating the time
        bin duration calculated from the vector of time bins.
        Default is 3, i.e. assumes milliseconds is the last significant digit.
    freqbins_key : str
        Key for accessing vector of frequency bins in files. Default is 'f'.
    timebins_key : str
        Key for accessing vector of time bins in files. Default is 't'.
    spect_key : str
        Key for accessing spectrogram in files. Default is 's'.
    audio_path_key : str
        Key for accessing path to source audio file for spectrogram in files.
        Default is 'audio_path'.

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

    Notes
    -----
    Each file should contain a spectrogram as a matrix and two vectors associated with it, a
    vector of frequency bins and time bins, where the values in those vectors are the values
    at the bin centers. (As far as vak is concerned, "vector" and "matrix" are synonymous with
    "array".)

    Since both mat files and npz files load into a dictionary-like structure,
    the arrays will be accessed with keys. By convention, these keys are 's', 'f', and 't'.
    If you use different keys you can let this function know by changing
    the appropriate arguments: spect_key, freqbins_key, timebins_key
    """
    # pre-conditions ---------------------------------------------------------------------------------------------------
    if spect_format not in constants.VALID_SPECT_FORMATS:
        raise ValueError(
            f"spect_format must be one of '{constants.VALID_SPECT_FORMATS}'; "
            f"format '{spect_format}' not recognized."
        )

    if all([arg is None for arg in (spect_dir, spect_files)]):
        raise ValueError("must specify one of: spect_dir, spect_files")

    if spect_dir and spect_files:
        raise ValueError(
            "received values for spect_dir and spect_files, unclear which to use"
        )

    if annot_list and annot_format is None:
        raise ValueError(
            "an annot_list was provided, but no annot_format was specified"
        )

    if annot_format is not None and annot_list is None:
        raise ValueError(
            "an annot_format was specified but no annot_list or spect_annot_map was provided"
        )

    if labelset is not None:
        labelset = labelset_to_set(labelset)

    # ---- get a list of spectrogram files + associated annotation files -----------------------------------------------
    if spect_dir:  # then get spect_files from that dir
        # note we already validated format above
        spect_files = sorted(pathlib.Path(spect_dir).glob(f"*{spect_format}"))

    if spect_files:  # (or if we just got them from spect_dir)
        if annot_list:
            spect_annot_map = map_annotated_to_annot(
                spect_files, annot_list, annot_format, annotated_ext=spect_ext
            )
        else:
            # no annotation, so map spectrogram files to None
            spect_annot_map = dict(
                (spect_path, None) for spect_path in spect_files
            )

    # use labelset if supplied, to filter
    if (
        labelset
    ):  # then assume user wants to filter out files where annotation has labels not in labelset
        for spect_path, annot in list(
            spect_annot_map.items()
        ):  # `list` so we can pop from dict without RuntimeError
            annot_labelset = set(annot.seq.labels)
            # below, set(labels_mapping) is a set of that dict's keys
            if not annot_labelset.issubset(set(labelset)):
                spect_annot_map.pop(spect_path)
                # because there's some label in labels that's not in labelset
                extra_labels = annot_labelset - set(labelset)
                logger.info(
                    f"Found labels, {extra_labels}, in {pathlib.Path(spect_path).name}, "
                    "that are not in labels_mapping. Skipping file.",
                )
                continue

    # ---- validate set of spectrogram files ---------------------------------------------------------------------------
    # regardless of whether we just made it or user supplied it
    spect_paths = list(spect_annot_map.keys())
    files.spect.is_valid_set_of_spect_files(
        spect_paths,
        spect_format,
        freqbins_key,
        timebins_key,
        spect_key,
        n_decimals_trunc,
    )

    # now that we have validated that duration of time bins is consistent across files, we can just open one file
    # to get that time bin duration. This way validation function has no side effects, like returning time bin, and
    # this is still relatively fast compared to looping through all files again
    timebin_dur = files.spect.timebin_dur(
        spect_paths[0], spect_format, timebins_key, n_decimals_trunc
    )

    # ---- actually make the dataframe ---------------------------------------------------------------------------------
    # this is defined here so all other arguments to 'to_dataframe' are in scope
    def _to_record(spect_annot_tuple):
        """helper function that enables parallelized creation
        of "records", i.e. rows for dataframe.
        Accepts a two-element tuple containing
        (1) a dictionary that represents a spectrogram
        and (2) annotation for that file"""
        spect_path, annot = spect_annot_tuple
        spect_path = pathlib.Path(spect_path)

        spect_dict = files.spect.load(spect_path, spect_format)

        spect_dur = spect_dict[spect_key].shape[-1] * timebin_dur
        if audio_path_key in spect_dict:
            audio_path = spect_dict[audio_path_key]
            if isinstance(audio_path, np.ndarray):
                # (because everything stored in .npz has to be in an ndarray)
                audio_path = audio_path.tolist()
        else:
            # try to figure out audio filename programmatically
            # if we can't, then we'll get back a None
            # (or an error)
            audio_path = files.spect.find_audio_fname(spect_path)

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
                abspath(spect_path),
                abspath(annot_path),
                (
                    annot_format
                    if annot_format
                    else constants.NO_ANNOTATION_FORMAT
                ),
                spect_dur,
                timebin_dur,
            ]
        )
        return record

    spect_path_annot_tuples = db.from_sequence(spect_annot_map.items())
    logger.info(
        "creating pandas.DataFrame representing dataset from spectrogram files",
    )
    with ProgressBar():
        records = list(spect_path_annot_tuples.map(_to_record))

    return pd.DataFrame.from_records(data=records, columns=DF_COLUMNS)
