"""Functions to parse toml config files."""
from __future__ import annotations

import pathlib

import tomlkit
import tomlkit.exceptions

from .config import Config
from .eval import EvalConfig
from .learncurve import LearncurveConfig
from .predict import PredictConfig
from .prep import PrepConfig
from .train import TrainConfig
from .validators import are_options_valid, are_tables_valid


TABLE_CLASSES = {
    "eval": EvalConfig,
    "learncurve": LearncurveConfig,
    "predict": PredictConfig,
    "prep": PrepConfig,
    "train": TrainConfig,
}

REQUIRED_OPTIONS = {
    "eval": [
        "checkpoint_path",
        "output_dir",
        "model",
    ],
    "learncurve": [
        "model",
        "root_results_dir",
    ],
    "predict": [
        "checkpoint_path",
        "model",
    ],
    "prep": [
        "data_dir",
        "output_dir",
    ],
    "train": [
        "model",
        "root_results_dir",
    ],
}


def _validate_tables_to_parse_arg_convert_list(tables_to_parse: str | list[str]) -> list[str]:
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
            table_name in list(TABLE_CLASSES.keys())
            for table_name in tables_to_parse
        ]
    ):
        raise ValueError(
            "All table names in 'tables_to_parse' should be valid names of tables. "
            f"Values for 'tables were: {tables_to_parse}.\n"
            f"Valid table names are: {list(TABLE_CLASSES.keys())}"
        )
    return tables_to_parse


def from_toml(
        config_dict: dict, toml_path: str | pathlib.Path | None = None, tables_to_parse: str | list[str] | None = None
        ) -> Config:
    """Load a TOML configuration file.

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
        instance of Config class, whose attributes correspond to
        tables in a config.toml file.
    """
    are_tables_valid(config_dict, toml_path)
    if tables_to_parse is None:
        tables_to_parse = list(
            TABLE_CLASSES.keys()
        )  # i.e., parse all tables
    else:
        tables_to_parse = _validate_tables_to_parse_arg_convert_list(tables_to_parse)

    config_kwargs = {}
    for table_name in tables_to_parse:
        if table_name in config_dict:
            are_options_valid(config_dict, table_name, toml_path)
            table_config_dict = config_dict[table_name]
            config_kwargs[table_name] = TABLE_CLASSES[table_name].from_config_dict(table_config_dict)
        else:
            raise KeyError(
                f"A table specified in `tables_to_parse` was not found in the config: {table_name}"
            )

    return Config(**config_kwargs)


def _load_toml_from_path(toml_path: str | pathlib.Path) -> dict:
    """Load a toml file from a path, and return as a :class:`dict`.

    Helper function to load toml config file,
    factored out to use in other modules when needed.
    Checks if ``toml_path`` exists before opening,
    and tries to give a clear message if an error occurs when loading."""
    toml_path = pathlib.Path(toml_path)
    if not toml_path.is_file():
        raise FileNotFoundError(f".toml config file not found: {toml_path}")

    try:
        with toml_path.open("r") as fp:
            config_dict: dict = tomlkit.load(fp)
    except tomlkit.exceptions.TOMLKitError as e:
        raise Exception(
            f"Error when parsing .toml config file: {toml_path}"
        ) from e

    if 'vak' not in config_dict:
        raise ValueError(
            "Toml file does not contain a top-level table named `vak`. "
            f"Please see example configuration files here: "
        )

    return config_dict['vak']


def from_toml_path(toml_path: str | pathlib.Path, tables_to_parse: list[str] | None = None) -> Config:
    """Parse a TOML configuration file and return as a :class:`Config`.

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
    config_dict: dict = _load_toml_from_path(toml_path)
    return from_toml(config_dict, toml_path, tables_to_parse)
