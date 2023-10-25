"""Helper functions used with frame classification datasets."""
from __future__ import annotations

import numpy as np

from ... import common
from . import constants


def sample_ids_array_filename_for_subset(subset: str) -> str:
    """Returns name of sample IDs array file for a subset of the training data."""
    return constants.SAMPLE_IDS_ARRAY_FILENAME.replace(
        ".npy", f"-{subset}.npy"
    )


def inds_in_sample_array_filename_for_subset(subset: str) -> str:
    """Returns name of inds in sample array file for a subset of the training data."""
    return constants.INDS_IN_SAMPLE_ARRAY_FILENAME.replace(
        ".npy", f"-{subset}.npy"
    )


def load_frames(frames_path, input_type):
    """Helper function that loads "frames",
    the input to the frame classification model.
    Loads audio or spectrogram, depending on
    :attr:`self.input_type`.
    This function assumes that audio is in wav format
    and spectrograms are in npz files.
    Also return ``frame_times``, either the time bins
    vector from a spectrogram file, or a vector
    the same length as the audio but where each sample
    number has been converted to seconds by dividing
    by the sampling rate.
    """
    if input_type == "audio":
        frames, samplerate = common.constants.AUDIO_FORMAT_FUNC_MAP[
            constants.FRAME_CLASSIFICATION_DATASET_AUDIO_FORMAT
        ](frames_path)
        frame_times = np.arange(frames.shape[-1]) / samplerate
    elif input_type == "spect":
        spect_dict = common.files.spect.load(frames_path)
        frames = spect_dict[common.constants.SPECT_KEY]
        frame_times = spect_dict[common.constants.TIMEBINS_KEY]
    return frames, frame_times
