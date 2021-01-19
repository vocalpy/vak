"""Utilities for input validation

adapted in part from scikit-learn under license
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py
"""
import warnings

import numpy as np


def column_or_1d(y, warn=False):
    """ravel column or 1d numpy array, else raise an error

    Parameters
    ----------
    y : array-like
    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array
    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def row_or_1d(y, warn=False):
    """ravel row or 1d numpy array, else raise an error

    Parameters
    ----------
    y : array-like
    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array
    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[0] == 1:
        if warn:
            warnings.warn("A row-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))
