"""Class that represents dataset table in configuration file."""

from __future__ import annotations

import pathlib

import attr.validators
from attr import asdict, define, field

from ..common.converters import expanded_user_path


@define
class DatasetConfig:
    """Class that represents dataset table in configuration file.

    Attributes
    ----------
    path : pathlib.Path
        Path to the directory that contains the dataset.
        Equivalent to the `root` parameter of :mod:`torchvision`
        datasets.
    splits_path : pathlib.Path, optional
        Path to file representing splits.
        Default is None.
    name : str, optional
        Name of dataset. Only required for built-in datasets
        from the :mod:`~vak.datasets` module. Default is None.
    params: dict, optional
        Parameters for dataset class,
        passed in as keyword arguments.
        E.g., ``window_size=2000``.
        Default is None.
    """

    path: pathlib.Path = field(converter=expanded_user_path)
    splits_path: pathlib.Path | None = field(
        converter=attr.converters.optional(expanded_user_path), default=None
    )
    name: str | None = field(
        converter=attr.converters.optional(str), default=None
    )
    params: dict | None = field(
        # we default to an empty dict instead of None
        # so we can still do **['dataset']['params'] everywhere we do when params are specified
        converter=attr.converters.optional(dict),
        default={},
    )

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> DatasetConfig:

        return cls(**config_dict)

    def asdict(self):
        """Convert this :class:`DatasetConfig` instance
        to a :class:`dict` that can be passed
        into functions that take a ``dataset_config`` argument,
        like :func:`vak.train` and :func:`vak.predict`.
        """
        return asdict(self)
