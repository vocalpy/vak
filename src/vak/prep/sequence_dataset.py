"""Helper functions for datasets annotated as sequences."""
from __future__ import annotations

import pathlib

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..common import annotation, files


def where_unlabeled(dataset_df: pd.DataFrame) -> npt.NDArray:
    """Returns a Boolean array that is True where
    an annotation in a dataset of annotated sequences
    has a sequence with unlabeled segments.

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        A dataframe representing a source dataset of audio signals
        or spectrograms, as returned by
        :func:`vak.prep.audio_dataset.prep_audio_dataset` or
        :func:`vak.prep.spectrogram_dataset.prep_spectrogram_dataset`.

    Returns
    -------
    where_unlabeled : numpy.ndarray
        Vector with Boolean dtype, where a True
        element indicates that the
        annotated sequence indexed by this has
        segments that are unlabeled.
    """
    annots = annotation.from_df(dataset_df)
    durations = dataset_df.duration.values

    has_unlabeled_list = []

    for annot, duration in zip(annots, durations):
        has_unlabeled_list.append(
            annotation.has_unlabeled(annot, duration)
        )

    return np.array(has_unlabeled_list).astype(bool)


def has_unlabeled(dataset_df: pd.DataFrame) -> bool:
    r"""Determine if a dataset of vocalization
    annotated as sequences has periods that are unlabeled.

    Used to decide whether an additional class needs to be added
    to the set of labels :math:`Y = {y_1, y_2, \dots, y_n}`,
    where the added class :math:`y_{n+1}`
    will represent the unlabeled "background" periods.

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        A dataframe representing a source dataset of audio signals
        or spectrograms, as returned by
        :func:`vak.prep.audio_dataset.prep_audio_dataset` or
        :func:`vak.prep.spectrogram_dataset.prep_spectrogram_dataset`.

    Returns
    -------
    has_unlabeled : bool
        If True, there are annotations in the dataset
        that have unlabeled periods.
    """
    return np.any(
        where_unlabeled(dataset_df)
    )
