"""Functions for input validation"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import numpy.typing as npt
import torch


def column_or_1d(y: npt.NDArray, warn: bool = False) -> npt.NDArray:
    """ravel column or 1d numpy array, else raise an error

    Parameters
    ----------
    y : array-like
    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array

    adapted in part from scikit-learn under license
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py
    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples, ), for example using ravel().",
                stacklevel=2,
            )
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
            warnings.warn(
                "A row-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples, ), for example using ravel().",
                stacklevel=2,
            )
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def is_a_directory(path):
    """check if given path is a directory"""
    return pathlib.Path(path).is_dir()


def is_a_file(path):
    """check if given path is a file"""
    return pathlib.Path(path).is_file()


def is_1d_tensor(t: torch.Tensor, name: str | None = None) -> bool:
    """Validate that input is a one-dimensional tensor
    
    Parameters
    ----------
    t : torch.Tensor

    Returns
    -------
    is_1d_tensor : bool
        Returns True if ``t`` is a one-dimensional tensor.
        If ``t`` is not a tensor, raise a TypeError, 
        and if ``t`` is not one-dimensional, 
        raises a ValueError.
    """
    if not isinstance(t, torch.Tensor):
        if name is not None:
            name_insert = f" of `{name}`"
        else:
            name_insert = ""
        raise TypeError(
            f"Expected type{name_insert} to be `torch.Tensor` but type was: {type(t)}"
        )

    if not t.ndim == 1:
        if name is not None:
            name_insert = f"`{name}` must be "
        else:
            name_insert = "Must be "
        raise ValueError(
            f"{name_insert}a 1-dimensional tensor but ndim={t.ndim}"
        )

    return True


def is_2d_tensor(t: torch.Tensor, name: str | None = None) -> bool:
    """Validate that input is a two-dimensional tensor
    
    Parameters
    ----------
    t : torch.Tensor

    Returns
    -------
    is_2d_tensor : bool
        Returns True if ``t`` is a two-dimensional tensor.
        If ``t`` is not a tensor, raise a TypeError, 
        and if ``t`` is not two-dimensional, 
        raises a ValueError.
    """
    if not isinstance(t, torch.Tensor):
        if name is not None:
            name_insert = f" of `{name}`"
        else:
            name_insert = ""
        raise TypeError(
            f"Expected type{name_insert} to be `torch.Tensor` but type was: {type(t)}"
        )

    if not t.ndim == 2:
        if name is not None:
            name_insert = f"`{name}` must be "
        else:
            name_insert = "Must be "
        raise ValueError(
            f"{name_insert}a 2-dimensional tensor but ndim={t.ndim}"
        )

    return True


def is_1d_or_2d_tensor(y: torch.Tensor, name: str | None = None) -> bool:
    """Validates that ``y`` is a 
    one-dimension or two-dimensional 
    :class:`torch.Tensor`.

    If ``y`` is not a :class:`torch.Tensor`,
    raises a TypeError.
    If ``y`` does not have one or two 
    dimensions, raises a ValueError.

    Parameters
    ----------
    y: torch.Tensor
        Array to be validated.
    name: str, optional
        Name of array in calling function.
        Used in any error message if supplied.

    Returns
    -------
    is_1d_or_2d_tensor: bool
        ``True`` if ``y.ndim==1 or y.ndim == 2``

    Examples
    --------
    >>> y = torch.tensor([[0, 1, 2], [0, 1, 2]])
    >>> vak.metrics.boundary_detection.validators.is_1d_or_2d_tensor(y)
    True

    >>> y = torch.tensor([0, 1, 2])
    >>> vak.metrics.boundary_detection.validators.is_1d_or_2d_tensor(y)
    True
    """
    if name:
        name += " "
    else:
        name = ""

    if not isinstance(y, torch.Tensor):
        raise TypeError(
            f"Input {name}should be a `torch.Tensor`, but type was: {type(y)}"
        )

    if y.ndim !=1 and y.ndim != 2:
        raise ValueError(
            f"Input {name}should be a one-dimensional or two-dimensional `torch.Tensor`, "
            f"but number of dimensions was: {y.ndim}"
        )
    return True


def is_non_negative(
    boundary_times: torch.FloatTensor, name: str | None = None
) -> bool:
    """Validates that ``y`` is a 
    :class:`torch.Tensor` with 
    all non-negative (>=0.0) values.

    Parameters
    ----------
    y: torch.Tensor
        Array to be validated.
    name: str, optional
        Name of array in calling function.
        Used in any error message if supplied.

    Returns
    -------
    is_non_negative: bool
        True if all values in ``y`` 
        are non-negative

    Examples
    --------
    >>> y = torch.tensor([0.0, 0.1, 0.,2])
    >>> vak.metrics.boundary_detection.validators.is_non_negative(y)
    True
    """
    if name:
        name += " "
    else:
        name = ""

    if not torch.all(boundary_times >= 0.0):
        raise ValueError(
            f"Values of boundaries tensor {name}must all be non-negative:\n{boundary_times}"
        )

    return True


def is_strictly_increasing(
    boundary_times: torch.FloatTensor, name: str | None = None
) -> bool:
    """Validates that ``y`` is a 
    :class:`torch.Tensor` with 
    strictly increasing values.

    Parameters
    ----------
    y: torch.Tensor
        Array to be validated.
    name: str, optional
        Name of array in calling function.
        Used in any error message if supplied.

    Returns
    -------
    is_strictly_increasing: bool
        ``True`` if 
        ``torch.all(y[1:] > y[:-1])`` 
        is ``True``.

    Examples
    --------
    >>> y = torch.tensor([0.0, 0.1, 0.,2])
    >>> vak.metrics.boundary_detection.validators.is_non_negative(y)
    True
    """
    if name:
        name += " "
    else:
        name = ""

    if boundary_times.numel() <= 1:
        # It's a valid boundary times tensor but there's no boundaries or just one boundary,
        # so we don't check that values are strictly increasing
        return True

    if not torch.all(boundary_times[1:] > boundary_times[:-1]):
        raise ValueError(
            f"Values of boundaries times {name}must be strictly increasing:\n{boundary_times}"
        )

    return True


def have_same_dtype(
    t1: torch.Tensor,
    t2: torch.Tensor,
    name1: str | None = None,
    name2: str | None = None,
) -> bool:
    """Validates that two tensors, ``t1`` and ``t2``, have the same :class:`~torch.dtype`.

    Parameters
    ----------
    t1 : torch.Tensor
        First tensor to be validated.
    t2 : torch.Tensor
        Second tensor to be validated.
    name1 : str, optional
        Name of first tensor in calling function.
        Used in any error message if both ``name1`` and ``name2`` are supplied.
    name2 : str, optional
        Name of second tensor in calling function.
        Used in any error message if both ``name1`` and ``name2`` are supplied.

    Returns
    -------
    have_same_dtype : bool
        True if ``arr1`` and ``arr2`` have the same :class:`~numpy.dtype`.
    """
    if not t1.dtype == t2.dtype:
        if name1 and name2:
            names = f"{name1} and {name2} "
        else:
            names = ""

        raise ValueError(
            f"Two tensors {names}must have the same dtype, but dtypes were: {t1.dtype} and {t2.dtype}"
        )

    return True


def have_same_ndim(
    t1: torch.Tensor,
    t2: torch.Tensor,
    name1: str | None = None,
    name2: str | None = None,
) -> bool:
    """Validates that two tensors, ``t1`` and ``t2``, have the same :prop:`torch.Tensor.ndim`.

    Parameters
    ----------
    t1 : torch.Tensor
        First tensor to be validated.
    t2 : torch.Tensor
        Second tensor to be validated.
    name1 : str, optional
        Name of first tensor in calling function.
        Used in any error message if both ``name1`` and ``name2`` are supplied.
    name2 : str, optional
        Name of second tensor in calling function.
        Used in any error message if both ``name1`` and ``name2`` are supplied.

    Returns
    -------
    have_same_ndim : bool
        True if ``arr1`` and ``arr2`` have the same :prop:`torch.Tensor.ndim`.
    """

    if not t1.ndim == t2.ndim:
        if name1 and name2:
            names = f"{name1} and {name2} "
        else:
            names = ""

        raise ValueError(
            f"Two tensors {names}must have the same number of dimensions, but t1.ndim={t1.ndim} and t2.ndim={t2.ndim}"
        )

    return True
