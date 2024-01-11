"""Functions for making a dataset of segments,
as used to train parametric UMAP and AVA models."""
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
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from dask.diagnostics import ProgressBar

from ...common import annotation, constants
from ...common.converters import expanded_user_path, labelset_to_set
from ...config.spect_params import SpectParamsConfig
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
    max_dur: float | None = None
) -> list[Segment]:
    """Get a list of :class:`Segment` instances, given
    the path to an audio file and an annotation that indicates
    where segments occur in that audio file.

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
    max_dur : float
        Maximum duration for segments.
        If a float value is specified,
        any segment with a duration larger than
        that value (in seconds) will be omitted
        from the returned list of segments.
        Default is None.

    Returns
    -------
    segments : list
        A :class:`list` of :class:`Segment` instances.

    Notes
    -----
    Function used by
    :func:`vak.prep.segment_dataset.prep_segment_dataset`.
    """
    data, samplerate = constants.AUDIO_FORMAT_FUNC_MAP[audio_format](
        audio_path
    )
    sample_dur = 1.0 / samplerate

    segments = []
    for segment_num, (onset_s, offset_s, label) in enumerate(zip(
        annot.seq.onsets_s, annot.seq.offsets_s, annot.seq.labels
    )):
        if max_dur is not None:
            segment_dur = offset_s - onset_s
            if segment_dur > max_dur:
                logger.info(
                    f"Segment {segment_num} in {pathlib.Path(audio_path).name}, "
                    f"with onset at {onset_s}s and offset at {offset_s}s with label '{label}',"
                    f"has duration ({segment_dur}) that is greater than "
                    f"maximum allowed duration ({max_dur})."
                    "Omitting segment from dataset."
                )
                continue
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
    segment: Segment,
    spect_params: SpectParamsConfig,
) -> npt.NDArray:
    """Compute a spectrogram given a :class:`Segment` instance.

    Parameters
    ----------
    segment : Segment
    spect_params : SpectParamsConfig


    Returns
    -------
    spect : numpy.ndarray

    Notes
    -----
    Function used by
    :func:`vak.prep.segment_dataset.prep_segment_dataset`.
    """
    data, samplerate = np.array(segment.data), segment.samplerate
    s, f, t = spectrogram(
        data,
        samplerate,
        spect_params.fft_size,
        spect_params.step_size,
        spect_params.thresh,
        spect_params.transform_type,
        spect_params.freq_cutoffs,
        spect_params.min_val,
        spect_params.max_val,
        spect_params.normalize,
    )

    return s, f, t


@attrs.define
class SpectToSave:
    """A spectrogram to be saved.

    Used by :func:`save_spect`.
    """

    spect: npt.NDArray
    f: npt.NDArray
    t: npt.NDArray
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
    npz_path : str
        Path to npz file containing spectrogram inside ``output_dir``
    """
    spect_dict = {
        "s": spect_to_save.spect,
        "f": spect_to_save.f,
        "t": spect_to_save.t,
    }

    basename = (
        os.path.basename(spect_to_save.audio_path)
        + f"-segment-{spect_to_save.ind}"
    )
    npz_path = os.path.join(
        os.path.normpath(output_dir), basename + ".spect.npz"
    )
    np.savez(npz_path, **spect_dict)
    return npz_path


def abspath(a_path):
    """Convert a path to an absolute path"""
    if isinstance(a_path, str) or isinstance(a_path, pathlib.Path):
        return str(pathlib.Path(a_path).absolute())
    elif np.isnan(a_path):
        return a_path


# ---- make spectrograms + records for dataframe -----------------------------------------------------------------------
@dask.delayed
def make_spect_return_record(
    segment: Segment,
    ind: int,
    spect_params: SpectParamsConfig,
    output_dir: pathlib.Path,
) -> tuple[tuple, int, float]:
    """Helper function that enables parallelized creation of "records",
    i.e. rows for dataframe, from .
    Accepts a two-element tuple containing (1) a dictionary that represents a spectrogram
    and (2) annotation for that file"""

    s, f, t = spectrogram_from_segment(
        segment,
        spect_params,
    )
    n_timebins = s.shape[-1]

    spect_to_save = SpectToSave(s, f, t, ind, segment.audio_path)
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

    return record, n_timebins, s.mean()


@dask.delayed
def pad_spectrogram(record: tuple, pad_length: float, padval: float = 0.) -> None:
    """Pads a spectrogram to a specified length on the left and right sides.

    Spectrogram is saved again after padding.

    Parameters
    ----------
    record : tuple
        Returned by :func:`make_spect_return_record`,
        has path to spectrogram file.
    pad_length : int
        Length to which spectrogram should be padded.

    Returns
    -------
    shape : tuple
        Shape of spectrogram after padding.
    """
    spect_path = record[0]  # 'spect_path'
    spect_dict = np.load(spect_path)
    spect = spect_dict["s"]

    excess_needed = pad_length - spect.shape[-1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    spect_padded = np.pad(
        spect, [(0, 0), (pad_left, pad_right)], "constant", constant_values=padval
    )
    new_spect_path = str(spect_path).replace(".npz", ".npy")
    np.save(new_spect_path, spect_padded)
    return new_spect_path, spect_padded.shape


@dask.delayed
def interp_spectrogram(
    record: tuple,
    max_dur: float,
    target_shape: tuple[int, int],
    normalize: bool = True,
    fill_value: float = 0.
):
    """Linearly interpolate a spectrogram to a target shape.

    Spectrogram is saved again after interpolation.

    Uses :func:`scipy.interpolate.RegularGridInterpolator`
    to treat the spectrogram as if it were a function of the
    frequencies vector :math:`f` and the times vector :math:`t`,
    then interpolates given new frequencies and times
    with the same range but with the number of values
    specified by the argument ``target_shape``.

    Parameters
    ----------
    record : tuple
        Returned by :func:`make_spect_return_record`,
        has path to spectrogram file.
    max_dur : float
        Maximum duration for segments.
        Used with ``target_shape`` when reshaping
        the spectrogram via interpolation.
        Default is None.
    target_shape : tuple
        Of ints, (target number of frequency bins,
        target number of time bins).
        Spectrograms of units will be reshaped
        by interpolation to have the specified
        number of frequency and time bins.
        The transformation is only applied if both this
        parameter and ``max_dur`` are specified.
        Default is None.
    normalize : bool
        If True, min-max normalize the spectrogram.
        Default is True.

    Returns
    -------
    shape : tuple
        Shape of spectrogram after interpolation.
    """
    spect_path = record[0]  # 'spect_path'
    spect_dict = np.load(spect_path)
    s = spect_dict["s"]
    f = spect_dict["f"]
    t = spect_dict["t"]

    # if max_dur and target_shape are specified we interpolate spectrogram to target shape, like AVA
    target_freqs = np.linspace(f.min(), f.max(), target_shape[0])
    duration = t.max() - t.min()
    new_duration = np.sqrt(duration * max_dur)  # stretched duration
    shoulder = 0.5 * (max_dur - new_duration)
    target_times = np.linspace(t.min() - shoulder, t.max() + shoulder, target_shape[1])
    ttnew, ffnew = np.meshgrid(target_times, target_freqs, indexing='ij', sparse=True)
    r = RegularGridInterpolator((t, f), s.T, bounds_error=False, fill_value=fill_value)
    s = r((ttnew, ffnew)).T
    if normalize:
        s_max, s_min = s.max(), s.min()
        s = (s - s_min) / (s_max - s_min)
        s = np.clip(s, 0.0, 1.0)
    new_spect_path = str(spect_path).replace(".npz", ".npy")
    np.save(new_spect_path, s)
    return new_spect_path, s.shape


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


def prep_segment_dataset(
    audio_format: str,
    output_dir: str | pathlib.Path,
    spect_params: SpectParamsConfig,
    data_dir: str | pathlib.Path,
    annot_format: str | None = None,
    annot_file: str | pathlib.Path | None = None,
    labelset: set | None = None,
    context_s: float = 0.005,
    max_dur: float | None = None,
    target_shape: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, tuple[int]]:
    """Prepare a dataset of segments.

    Finds segments with a segmenting algorithm,
    then computes a spectrogram for each segment
    and saves in npy files.
    Finally, assigns each npy file to a split
    and moves files into split directories
    inside the directory representing the dataset.

    Parameters
    ----------
    audio_format : str
        Format of audio files. One of {'wav', 'cbin'}.
        Default is ``None``, but either ``audio_format`` or ``spect_format``
        must be specified.
    output_dir : str
        Path to location where data sets should be saved.
        Default is ``None``, in which case it defaults to ``data_dir``.
    spect_params : dict, vak.config.SpectParams
        Parameters for creating spectrograms. Default is ``None``.
    data_dir : str, pathlib.Path
        Path to directory with files from which to make dataset.
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
    context_s : float
        Number of seconds of "context" around unit to
        add, i.e., time before and after the onset
        and offset respectively. Default is 0.005s,
        5 milliseconds.
    max_dur : float
        Maximum duration for segments.
        If a float value is specified,
        any segment with a duration larger than
        that value (in seconds) will be omitted
        from the dataset. Default is None.
    target_shape : tuple
        Of ints, (target number of frequency bins,
        target number of time bins).
        Spectrograms of units will be reshaped
        by interpolation to have the specified
        number of frequency and time bins.
        The transformation is only applied if both this
        parameter and ``max_dur`` are specified.
        Default is None.

    Returns
    -------
    segment_df : pandas.DataFrame
        A DataFrame representing all the segments in the dataset.
    shape: tuple
        A tuple representing the shape of all spectrograms in the dataset.
        The spectrograms of all segments are padded so that they are all
        as wide as the widest segment (i.e, the one with the longest duration).
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
            audio_path, annot, audio_format, context_s, max_dur
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

    # ---- make and save all spectrograms *before* interpolating or padding
    # This is a design choice to avoid keeping all the spectrograms in memory
    # but since we want to pad all spectrograms to be the same width,
    # it requires us to go back, load each one, and pad it.
    # Might be worth looking at how often typical dataset sizes in memory and whether this is really necessary.
    records_n_timebins_tuples = []
    for ind, segment in enumerate(segments):
        records_n_timebins_tuple = make_spect_return_record(
            segment, ind, spect_params, output_dir,
        )
        records_n_timebins_tuples.append(records_n_timebins_tuple)
    with ProgressBar():
        records_n_timebins_tuples: list[tuple[tuple, int]] = dask.compute(
            *records_n_timebins_tuples
        )

    # we use n_timebins to pad to the same length,
    # and spect_means to fill with the mean across all spectrograms
    # when we interpolate
    records, n_timebins_list = [], []
    for records_n_timebins_tuple in records_n_timebins_tuples:
        record, n_timebins, spect_mean = records_n_timebins_tuple
        records.append(record)
        n_timebins_list.append(n_timebins)

    # ---- either interpolate or pad spectrograms so they are all the same size
    fill_value = spect_params.min_val if spect_params.min_val else 0.

    if max_dur is not None and target_shape is not None:
        interpolated = []
        for record in records:
            interpolated.append(
                interp_spectrogram(
                    record, max_dur, target_shape, spect_params.normalize, fill_value
                ))
        with ProgressBar():
            path_shape_tuples = dask.compute(*interpolated)

    else:
        # then we pad
        pad_length = max(n_timebins_list)

        padded = []
        for record in records:
            padded.append(pad_spectrogram(record, pad_length, padval=fill_value))
        with ProgressBar():
            path_shape_tuples = dask.compute(*padded)

    # ---- clean up npz files with spectrograms, don't need anymore
    npz_files = sorted(output_dir.glob('*npz'))
    for npz_file in npz_files:
        npz_file.unlink()

    paths, shapes = [], []
    for path, shape in path_shape_tuples:
        paths.append(path)
        shapes.append(shape)
    shape = set(shapes)
    assert (
        len(shape) == 1
    ), f"Did not find a single unique shape for all spectrograms. Instead found: {shape}"
    shape = shape.pop()

    new_records = []
    for record, path in zip(records, paths):
        new_records.append(
            tuple(
                [path, *record[1:]]
            )
        )
    segment_df = pd.DataFrame.from_records(new_records, columns=DF_COLUMNS)

    return segment_df, shape
