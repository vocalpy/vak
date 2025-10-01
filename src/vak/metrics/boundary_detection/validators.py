import torch


def is_1d_tensor(y: torch.Tensor, name: str | None = None) -> bool:
    """Validates that ``y`` is a
    one-dimensional :class:`torch.Tensor`.

    Parameters
    ----------
    y: torch.Tensor
        Array to be validated.
    name: str, optional
        Name of array in calling function.
        Used in any error message if supplied.

    Returns
    -------
    is_1d_tensor: bool
        True if ``y`` is one-dimensional.

    Examples
    --------
    >>> y = torch.tensor([0, 1, 2])
    >>> vak.metrics.boundary_detection.validators.is_1d_tensor(y)
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

    if y.ndim != 1:
        raise ValueError(
            f"Input {name}should be a one-dimensional `torch.Tensor`, "
            f"but number of dimensions was: {y.ndim}"
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
