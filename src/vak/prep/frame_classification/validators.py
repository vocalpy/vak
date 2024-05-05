"""Validators for frame classification datasets"""

from __future__ import annotations

import pandas as pd


def validate_and_get_frame_dur(
    dataset_df: pd.DataFrame, input_type: str
) -> float:
    """Validate that there is a single, unique value for the
    frame duration for all samples (audio signals / spectrograms)
    in a dataset. If so, return that value.

    The dataset is represented as a pandas DataFrame.

    Parameters
    ----------
    dataset_df : pandas.Dataframe
        A pandas.DataFrame created by
        :func:`vak.prep.spectrogram_dataset.prep_spectrogram_dataset`
        or :func:`vak.prep.audio_dataset.prep_audio_dataset`.
    input_type : str
        The type of input to the neural network model.
        One of {'audio', 'spect'}.

    Returns
    -------
    frame_dur : float
        The duration of a time bin in seconds
        for all spectrograms in the dataset.
    """
    from .. import constants  # avoid circular import

    if input_type not in constants.INPUT_TYPES:
        raise ValueError(
            f"``input_type`` must be one of: {constants.INPUT_TYPES}\n"
            f"Value for ``input_type`` was: {input_type}"
        )

    # TODO: handle possible KeyError here?
    if input_type == "audio":
        frame_dur = dataset_df["sample_dur"].unique()
    elif input_type == "spect":
        frame_dur = dataset_df["timebin_dur"].unique()

    if len(frame_dur) > 1:
        raise ValueError(
            f"Found more than one frame duration in dataset: {frame_dur}"
        )

    frame_dur = frame_dur.item()

    return frame_dur
