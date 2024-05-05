"""Class that represents the TOML configuration file used with the vak command-line interface."""

from __future__ import annotations

import pathlib

from attr.validators import instance_of, optional
from attrs import define, field

from . import load
from .eval import EvalConfig
from .learncurve import LearncurveConfig
from .predict import PredictConfig
from .prep import PrepConfig
from .train import TrainConfig
from .validators import are_keys_valid, are_tables_valid

TABLE_CLASSES_MAP = {
    "eval": EvalConfig,
    "learncurve": LearncurveConfig,
    "predict": PredictConfig,
    "prep": PrepConfig,
    "train": TrainConfig,
}


def _validate_tables_to_parse_arg_convert_list(
    tables_to_parse: str | list[str],
) -> list[str]:
    """Helper function used by :func:`from_toml` that
    validates the ``tables_to_parse`` argument,
    and returns it as a list of strings."""
    if isinstance(tables_to_parse, str):
        tables_to_parse = [tables_to_parse]

    if not isinstance(tables_to_parse, list):
        raise TypeError(
            f"`tables_to_parse` should be a string or list of strings but type was: {type(tables_to_parse)}"
        )

    if not all(
        [isinstance(table_name, str) for table_name in tables_to_parse]
    ):
        raise ValueError(
            "All table names in 'tables_to_parse' should be strings"
        )
    if not all(
        [
            table_name in list(TABLE_CLASSES_MAP.keys())
            for table_name in tables_to_parse
        ]
    ):
        raise ValueError(
            "All table names in 'tables_to_parse' should be valid names of tables. "
            f"Values for 'tables were: {tables_to_parse}.\n"
            f"Valid table names are: {list(TABLE_CLASSES_MAP.keys())}"
        )
    return tables_to_parse


@define
class Config:
    """Class that represents the TOML configuration file used with the vak command-line interface.

    Attributes
    ----------
    prep : vak.config.prep.PrepConfig
        Represents ``[vak.prep]`` table of config.toml file
    train : vak.config.train.TrainConfig
        Represents ``[vak.train]`` table of config.toml file
    eval : vak.config.eval.EvalConfig
        Represents ``[vak.eval]`` table of config.toml file
    predict : vak.config.predict.PredictConfig
        Represents ``[vak.predict]`` table of config.toml file.
    learncurve : vak.config.learncurve.LearncurveConfig
        Represents ``[vak.learncurve]`` table of config.toml file
    """

    prep = field(validator=optional(instance_of(PrepConfig)), default=None)
    train = field(validator=optional(instance_of(TrainConfig)), default=None)
    eval = field(validator=optional(instance_of(EvalConfig)), default=None)
    predict = field(
        validator=optional(instance_of(PredictConfig)), default=None
    )
    learncurve = field(
        validator=optional(instance_of(LearncurveConfig)), default=None
    )

    @classmethod
    def from_config_dict(
        cls,
        config_dict: dict,
        tables_to_parse: str | list[str] | None = None,
        toml_path: str | pathlib.Path | None = None,
    ) -> "Config":
        """Return instance of :class:`Config` class,
        given a :class:`dict` containing the contents of
        a TOML configuration file.

        This :func:`classmethod` expects the output
        of :func:`vak.config.load._load_from_toml_path`,
        that converts a :class:`tomlkit.TOMLDocument`
        to a :class:`dict`, and returns the :class:`dict`
        that is accessed by the top-level key ``'vak'``.

        Parameters
        ----------
        config_dict : dict
            Python ``dict`` containing a .toml configuration file,
            parsed by the ``toml`` library.
        toml_path : str, pathlib.Path
            path to a configuration file in TOML format. Default is None.
            Not required, used only to make any error messages clearer.
        tables_to_parse : str, list
            Name of top-level table or tables from configuration
            file that should be parsed. Can be a string
            (single table) or list of strings (multiple
            tables). Default is None,
            in which case all are validated and parsed.

        Returns
        -------
        config : vak.config.parse.Config
            instance of :class:`Config` class,
            whose attributes correspond to the
            top-level tables in a config.toml file.
        """
        are_tables_valid(config_dict, toml_path)
        if tables_to_parse is None:
            tables_to_parse = list(
                config_dict.keys()
            )  # i.e., parse all top-level tables
        else:
            tables_to_parse = _validate_tables_to_parse_arg_convert_list(
                tables_to_parse
            )

        config_kwargs = {}
        for table_name in tables_to_parse:
            if table_name in config_dict:
                are_keys_valid(config_dict, table_name, toml_path)
                table_config_dict = config_dict[table_name]
                config_kwargs[table_name] = TABLE_CLASSES_MAP[
                    table_name
                ].from_config_dict(table_config_dict)
            else:
                raise KeyError(
                    f"A table specified in `tables_to_parse` was not found in the config: {table_name}"
                )

        return cls(**config_kwargs)

    @classmethod
    def from_toml_path(
        cls,
        toml_path: str | pathlib.Path,
        tables_to_parse: list[str] | None = None,
    ) -> "Config":
        """Return instance of :class:`Config` class,
        given the path to a TOML configuration file.

        Parameters
        ----------
        toml_path : str, pathlib.Path
            Path to a configuration file in TOML format.
            Parsed by ``toml`` library, then converted to an
            instance of ``vak.config.parse.Config`` by
            calling ``vak.parse.from_toml``
        tables_to_parse : str, list
            Name of table or tables from configuration
            file that should be parsed. Can be a string
            (single table) or list of strings (multiple
            tables). Default is None,
            in which case all are validated and parsed.

        Returns
        -------
        config : vak.config.parse.Config
            instance of :class:`Config` class, whose attributes correspond to
            tables in a config.toml file.
        """
        config_dict: dict = load._load_toml_from_path(toml_path)
        return cls.from_config_dict(config_dict, tables_to_parse, toml_path)
