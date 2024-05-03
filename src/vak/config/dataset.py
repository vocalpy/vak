"""Class that represents dataset table in configuration file."""
from __future__ import annotations

import pathlib

from attr import define, field
import attr.validators


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
    name : str, optional
        Name of dataset. Only required for built-in datasets
        from the :mod:`~vak.datasets` module.
    """
    path: pathlib.Path = field(converter=pathlib.Path)
    splits_path: pathlib.Path | None = field(
        converter=attr.converters.optional(pathlib.Path), default=None
        )
    name: str | None = field(
        converter=attr.converters.optional(str), default=None
        )

    @classmethod
    def from_config_dict(cls, dict_: dict) -> DatasetConfig:
        return cls(
            path=dict_.get('path'),
            splits_path=dict_.get('splits_path'),
            name=dict_.get('name'),
        )