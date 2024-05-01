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
    name : str, optional
        Name of dataset. Only required for built-in datasets
        from the :mod:`~vak.datasets` module.
    path : pathlib.Path
        Path to the directory that contains the dataset.
        Equivalent to the `root` parameter of :module:`torchvision`
        datasets.
    splits_path : pathlib.Path, optional
        Path to file representing splits.
    """
    path: pathlib.Path = field(converter=pathlib.Path)
    name: str | None = field(
        converter=attr.converters.optional(str), default=None
        )
    splits_path: pathlib.Path | None = field(
        converter=attr.converters.optional(pathlib.Path), default=None
        )

    @classmethod
    def from_config_dict(cls, dict_: dict) -> DatasetConfig:
        return cls(
            path=dict_.get('path'),
            name=dict_.get('name'),
            splits_path=dict_.get('splits_path')
        )
