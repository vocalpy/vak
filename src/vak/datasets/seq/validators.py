"""Validators for sequence datasets"""
import numpy as np
import pandas as pd

from ... import annotation, files


def where_unlabeled(csv_path, timebins_key="t"):
    """Find where there are vocalizations in a dataset
    annotated as sequences that have unlabeled periods
    in those annotations.

    Parameters
    ----------
    csv_path : str, Path
        Path to .csv file representing dataset.
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
    vak_df = pd.read_csv(csv_path)
    annots = annotation.from_df(vak_df)

    has_unlabeled_list = []
    for annot, spect_path in zip(annots, vak_df["spect_path"].values):
        spect_dict = files.spect.load(spect_path)
        timebins = spect_dict[timebins_key]
        duration = timebins[-1]

        has_unlabeled_list.append(
            annotation.has_unlabeled(annot, duration)
        )

    return np.array(has_unlabeled_list)


def has_unlabeled(csv_path, timebins_key="t"):
    r"""Determine if a dataset of vocalization
    annotated as sequences has periods that are unlabeled.

    Used to decide whether an additional class needs to be added
    to the set of labels :math:`Y = {y_1, y_2, \dots, y_n}`,
    where the added class :math:`y_{n+1}`
    will represent the unlabeled "background" periods.

    Parameters
    ----------
    csv_path : str, Path
        Path to .csv file representing dataset.
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
        where_unlabeled(csv_path, timebins_key)
    )
