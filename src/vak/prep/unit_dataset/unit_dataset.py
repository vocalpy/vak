"""Functions for making a dataset of units from sequences,
as used to train dimensionality reduction models."""

from __future__ import annotations

import logging
import os
import pathlib

import attrs
import crowsetta
import dask
import dask.delayed
import numpy as np
import numpy.typing as npt
import pandas as pd
from dask.diagnostics import ProgressBar

from ...common import annotation, constants
from ...common.converters import expanded_user_path, labelset_to_set
from ..spectrogram_dataset.audio_helper import files_from_dir
from ..spectrogram_dataset.spect import spectrogram

logger = logging.getLogger(__name__)


@attrs.define
class Segment:
    """Dataclass that represents a segment
    from segmented audio or spectrogram.

    The attributes are metadata used to track
    the origin of this segment in a dataset
    of such segments.

    The dataset including metadata is saved as a csv file
    where these attributes become the columns.
    """

    data: npt.NDArray
    samplerate: int
    onset_s: float
    offset_s: float
    label: str
    sample_dur: float
    segment_dur: float
    audio_path: str
    annot_path: str


@dask.delayed
def get_segment_list(
    audio_path: str,
    annot: crowsetta.Annotation,
    audio_format: str,
    context_s: float = 0.005,
) -> list[Segment]:
    """Get a list of :class:`Segment` instances, given
    the path to an audio file and an annotation that indicates
    where segments occur in that audio file.

    Function used by
    :func:`vak.prep.dimensionality_reduction.unit_dataset.prep_unit_dataset`.

    Parameters
    ----------
    audio_path : str
        Path to an audio file.
    annot : crowsetta.Annotation
        Annotation for audio file.
    audio_format : str
        String representing audio file format, e.g. 'wav'.
    context_s : float
        Number of seconds of "context" around unit to
        add, i.e., time before and after the onset
        and offset respectively. Default is 0.005s,
        5 milliseconds.

    Returns
    -------
    segments : list
        A :class:`list` of :class:`Segment` instances.
    """
    data, samplerate = constants.AUDIO_FORMAT_FUNC_MAP[audio_format](
        audio_path
    )
    sample_dur = 1.0 / samplerate

    segments = []
    for onset_s, offset_s, label in zip(
        annot.seq.onsets_s, annot.seq.offsets_s, annot.seq.labels
    ):
        onset_s -= context_s
        offset_s += context_s
        onset_ind = int(np.floor(onset_s * samplerate))
        offset_ind = int(np.ceil(offset_s * samplerate))
        segment_data = data[onset_ind : offset_ind + 1]  # noqa: E203
        segment_dur = segment_data.shape[-1] * sample_dur
        segment = Segment(
            segment_data,
            samplerate,
            onset_s,
            offset_s,
            label,
            sample_dur,
            segment_dur,
            audio_path,
            annot.annot_path,
        )
        segments.append(segment)

    return segments


def spectrogram_from_segment(
    segment: Segment, spect_params: dict
) -> npt.NDArray:
    """Compute a spectrogram given a :class:`Segment` instance.

    Parameters
    ----------
    segment : Segment
    spect_params : dict

    Returns
    -------
    spect : numpy.ndarray
    """
    data, samplerate = np.array(segment.data), segment.samplerate
    s, _, _ = spectrogram(
        data,
        samplerate,
        spect_params.fft_size,
        spect_params.step_size,
        spect_params.thresh,
        spect_params.transform_type,
        spect_params.freq_cutoffs,
    )
    return s


@attrs.define
class SpectToSave:
    """A spectrogram to be saved.

    Used by :func:`save_spect`.
    """

    spect: npt.NDArray
    ind: int
    audio_path: str


def save_spect(
    spect_to_save: SpectToSave, output_dir: str | pathlib.Path
) -> str:
    """Save a spectrogram array to an npy file.

    The filename is build from the attributes of ``spect_to_save``,
    saved in output dir, and the full path is returned as a string.

    Parameters
    ----------
    spect_to_save : SpectToSave
    output_dir : str, pathlib.Path

    Returns
    -------
    npy_path : str
        Path to npy file containing spectrogram inside ``output_dir``
    """
    basename = (
        os.path.basename(spect_to_save.audio_path)
        + f"-segment-{spect_to_save.ind}"
    )
    npy_path = os.path.join(
        os.path.normpath(output_dir), basename + ".spect.npy"
    )
    np.save(npy_path, spect_to_save.spect)
    return npy_path


def abspath(a_path):
    """Convert a path to an absolute path"""
    if isinstance(a_path, str) or isinstance(a_path, pathlib.Path):
        return str(pathlib.Path(a_path).absolute())
    elif np.isnan(a_path):
        return a_path


# ---- make spectrograms + records for dataframe -----------------------------------------------------------------------
@dask.delayed
def make_spect_return_record(
    segment: Segment, ind: int, spect_params: dict, output_dir: pathlib.Path
) -> tuple:
    """Helper function that enables parallelized creation of "records",
    i.e. rows for dataframe, from .
    Accepts a two-element tuple containing (1) a dictionary that represents a spectrogram
    and (2) annotation for that file"""

    spect = spectrogram_from_segment(segment, spect_params)
    n_timebins = spect.shape[-1]

    spect_to_save = SpectToSave(spect, ind, segment.audio_path)
    spect_path = save_spect(spect_to_save, output_dir)
    record = tuple(
        [
            abspath(spect_path),
            abspath(segment.audio_path),
            abspath(segment.annot_path),
            segment.onset_s,
            segment.offset_s,
            segment.label,
            segment.samplerate,
            segment.sample_dur,
            segment.segment_dur,
        ]
    )

    return record, n_timebins


@dask.delayed
def pad_spectrogram(record: tuple, pad_length: float) -> None:
    """Pads a spectrogram to a specified length on the left and right sides.
    Spectrogram is saved again after padding.

    Parameters
    ----------
    record : tuple
    pad_length : int
    """
    spect_path = record[0]  # 'spect_path'
    spect = np.load(spect_path)

    excess_needed = pad_length - spect.shape[-1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    spect_padded = np.pad(
        spect, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0
    )
    np.save(spect_path, spect_padded)
    return spect_padded.shape


# constant, used for names of columns in DataFrame below
DF_COLUMNS = [
    "spect_path",
    "audio_path",
    "annot_path",
    "onset_s",
    "offset_s",
    "label",
    "samplerate",
    "sample_dur",
    "duration",
]


def prep_unit_dataset(
    audio_format: str,
    output_dir: str,
    spect_params: dict,
    data_dir: list | None = None,
    annot_format: str | None = None,
    annot_file: str | pathlib.Path | None = None,
    labelset: set | None = None,
    context_s: float = 0.005,
) -> pd.DataFrame:
    """Prepare a dataset of units from sequences,
    e.g., all syllables segmented out of a dataset of birdsong.

    Parameters
    ----------
    audio_format
    output_dir
    spect_params
    data_dir
    annot_format
    annot_file
    labelset
    context_s

    Returns
    -------
    unit_df : pandas.DataFrame
        A DataFrame representing all the units in the dataset.
    shape: tuple
        A tuple representing the shape of all spectograms in the dataset.
        The spectrograms of all units are padded so that they are all
        as wide as the widest unit (i.e, the one with the longest duration).
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
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir not found: {data_dir}")

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

    # use labelset, if supplied, with annotations, if any, to filter;
    if (
        labelset and annot_list
    ):  # then remove annotations with labels not in labelset
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

    segments = []
    for audio_path, annot in audio_annot_map.items():
        segment_list = dask.delayed(get_segment_list)(
            audio_path, annot, audio_format, context_s
        )
        segments.append(segment_list)

    logger.info(
        "Loading audio for all segments in all files",
    )
    with ProgressBar():
        segments: list[list[Segment]] = dask.compute(*segments)
    segments: list[Segment] = [
        segment for segment_list in segments for segment in segment_list
    ]

    # ---- make and save all spectrograms *before* padding
    # This is a design choice to avoid keeping all the spectrograms in memory
    # but since we want to pad all spectrograms to be the same width,
    # it requires us to go back, load each one, and pad it.
    # Might be worth looking at how often typical dataset sizes in memory and whether this is really necessary.
    records_n_timebins_tuples = []
    for ind, segment in enumerate(segments):
        records_n_timebins_tuple = make_spect_return_record(
            segment, ind, spect_params, output_dir
        )
        records_n_timebins_tuples.append(records_n_timebins_tuple)
    with ProgressBar():
        records_n_timebins_tuples: list[tuple[tuple, int]] = dask.compute(
            *records_n_timebins_tuples
        )

    records, n_timebins_list = [], []
    for records_n_timebins_tuple in records_n_timebins_tuples:
        record, n_timebins = records_n_timebins_tuple
        records.append(record)
        n_timebins_list.append(n_timebins)

    pad_length = max(n_timebins_list)

    padded = []
    for record in records:
        padded.append(pad_spectrogram(record, pad_length))
    with ProgressBar():
        shapes: list[tuple[int, int]] = dask.compute(*padded)

    shape = set(shapes)
    assert (
        len(shape) == 1
    ), f"Did not find a single unique shape for all spectrograms. Instead found: {shape}"
    shape = shape.pop()

    unit_df = pd.DataFrame.from_records(records, columns=DF_COLUMNS)

    return unit_df, shape
