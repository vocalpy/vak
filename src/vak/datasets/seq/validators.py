"""Validators for sequence datasets"""
from __future__ import annotations

import pathlib

import numpy as np
import numpy.typing as npt
import pandas as pd

from ... import annotation, files


def where_unlabeled(dataset_csv_path: str | pathlib.Path,
                    timebins_key: str = "t") -> npt.NDArray:
    """Find where there are vocalizations in a dataset
    annotated as sequences that have unlabeled periods
    in those annotations.

    Parameters
    ----------
    dataset_csv_path : str, Path
        Path to csv file representing dataset.
        Saved by ``vak prep`` in the root of a directory
        that contains the prepared dataset.
    timebins_key : str
        Key used to access timebins vector in spectrogram files.
        Default is 't'.

    Returns
    -------
    where_unlabeled : numpy.ndarray
        Vector with Boolean dtype, where a True
        element indicates that the
        vocalization indexed by this has
        periods that are unlabeled by the annotations.
    """
    dataset_csv_path = pathlib.Path(dataset_csv_path)
    # next line, we assume csv path is in root of dataset dir, so we can use parent for dataset path arg
    dataset_path = dataset_csv_path.parent
    dataset_df = pd.read_csv(dataset_csv_path)
    annots = annotation.from_df(dataset_df, dataset_path=dataset_path)

    has_unlabeled_list = []
    for annot, spect_path in zip(annots, dataset_df["spect_path"].values):
        spect_dict = files.spect.load(dataset_path / spect_path)
        timebins = spect_dict[timebins_key]
        duration = timebins[-1]

        has_unlabeled_list.append(
            annotation.has_unlabeled(annot, duration)
        )

    return np.array(has_unlabeled_list)


def has_unlabeled(dataset_csv_path: str | pathlib.Path,
                  timebins_key: str = "t" ) -> bool:
    r"""Determine if a dataset of vocalization
    annotated as sequences has periods that are unlabeled.

    Used to decide whether an additional class needs to be added
    to the set of labels :math:`Y = {y_1, y_2, \dots, y_n}`,
    where the added class :math:`y_{n+1}`
    will represent the unlabeled "background" periods.

    Parameters
    ----------
    dataset_csv_path : str, Path
        Path to csv file representing dataset.
        Saved by ``vak prep`` in the root of a directory
        that contains the prepared dataset.
    timebins_key : str
        Key used to access timebins vector in spectrogram files.
        Default is 't'.

    Returns
    -------
    has_unlabeled : bool
        If True, there are annotations in the dataset
        that have unlabeled periods.
    """
    return np.any(
        where_unlabeled(dataset_csv_path, timebins_key)
    )
