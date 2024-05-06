"""validators used by attrs-based classes and by vak.parse.parse_config"""

import pathlib

import tomlkit

from .. import models
from ..common import constants


def is_a_directory(instance, attribute, value):
    """check if given path is a directory"""
    if not pathlib.Path(value).is_dir():
        raise NotADirectoryError(
            f"Value specified for {attribute.name} of {type(instance)} not recognized as a directory:\n"
            f"{value}"
        )


def is_a_file(instance, attribute, value):
    """check if given path is a file"""
    if not pathlib.Path(value).is_file():
        raise FileNotFoundError(
            f"Value specified for {attribute.name} of {type(instance)} not recognized as a file:\n"
            f"{value}"
        )


def is_valid_model_name(instance, attribute, value: str) -> None:
    """Validate model name."""
    if value not in models.registry.MODEL_NAMES:
        raise ValueError(
            f"Invalid model name: {value}.\nValid model names are: {models.registry.MODEL_NAMES}"
        )


def is_audio_format(instance, attribute, value):
    """Check if valid audio format"""
    if value not in constants.VALID_AUDIO_FORMATS:
        raise ValueError(f"{value} is not a valid format for audio files")


def is_annot_format(instance, attribute, value):
    """Check if valid annotation format"""
    if value not in constants.VALID_ANNOT_FORMATS:
        raise ValueError(
            f"{value} is not a valid format for annotation files.\n"
            f"Valid formats are: {constants.VALID_ANNOT_FORMATS}"
        )


def is_spect_format(instance, attribute, value):
    """Check if valid format for spectrograms"""
    if value not in constants.VALID_SPECT_FORMATS:
        raise ValueError(
            f"{value} is not a valid format for spectrogram files.\n"
            f"Valid formats are: {constants.VALID_SPECT_FORMATS}"
        )


CONFIG_DIR = pathlib.Path(__file__).parent
VALID_TOML_PATH = CONFIG_DIR.joinpath("valid-version-1.2.toml")
with VALID_TOML_PATH.open("r") as fp:
    VALID_DICT = tomlkit.load(fp)["vak"]
VALID_TOP_LEVEL_TABLES = list(VALID_DICT.keys())
VALID_KEYS = {
    table_name: list(table_config_dict.keys())
    for table_name, table_config_dict in VALID_DICT.items()
}


def are_tables_valid(config_dict, toml_path=None):
    """Validate top-level tables in class:`dict`.

    This function expects the ``config_dict``
    returned by :func:`vak.config.load._load_from_toml_path`.
    """
    tables = list(config_dict.keys())
    from ..cli.cli import CLI_COMMANDS  # avoid circular import

    cli_commands_besides_prep = [
        command for command in CLI_COMMANDS if command != "prep"
    ]
    tables_that_are_commands_besides_prep = [
        table for table in tables if table in cli_commands_besides_prep
    ]
    if len(tables_that_are_commands_besides_prep) == 0:
        raise ValueError(
            "Did not find a table related to a vak command in config besides `prep`.\n"
            f"Sections in config were: {tables}\n"
            "Please see example toml configuration files here: https://github.com/vocalpy/vak/tree/main/doc/toml"
        )

    if len(tables_that_are_commands_besides_prep) > 1:
        raise ValueError(
            "Found multiple tables related to a vak command in config besides `prep`.\n"
            f"Those tables are: {tables_that_are_commands_besides_prep}. "
            f"Please use just one command besides `prep` per .toml configuration file.\n"
            "See example toml configuration files here: https://github.com/vocalpy/vak/tree/main/doc/toml"
        )

    for table in tables:
        if table not in VALID_TOP_LEVEL_TABLES:
            if toml_path:
                err_msg = (
                    f"Top-level table defined in {toml_path} is not valid: {table}\n"
                    f"Valid top-level tables are: {VALID_TOP_LEVEL_TABLES}\n"
                    "Please see example toml configuration files here: "
                    "https://github.com/vocalpy/vak/tree/main/doc/toml"
                )
            else:
                err_msg = (
                    f"Table defined in toml config is not valid: {table}\n"
                    f"Valid top-level tables are: {VALID_TOP_LEVEL_TABLES}\n"
                    "Please see example toml configuration files here: "
                    "https://github.com/vocalpy/vak/tree/main/doc/toml"
                )
            raise ValueError(err_msg)


def are_keys_valid(
    config_dict: dict,
    table_name: str,
    toml_path: str | pathlib.Path | None = None,
) -> None:
    """Given a :class:`dict` containing the *entire* configuration loaded from a toml file,
    validate the key names for a specific top-level table, e.g. ``vak.train`` or ``vak.predict``
    """
    table_keys = set(config_dict[table_name].keys())
    valid_keys = set(VALID_KEYS[table_name])
    if not table_keys.issubset(valid_keys):
        invalid_keys = table_keys - valid_keys
        if toml_path:
            err_msg = (
                f"The following keys from '{table_name}' table in "
                f"the config file '{toml_path.name}' are not valid:\n{invalid_keys}"
            )
        else:
            err_msg = (
                f"The following keys from '{table_name}' table in "
                f"the toml config are not valid:\n{invalid_keys}"
            )
        raise ValueError(err_msg)


def are_table_keys_valid(
    table_config_dict: dict,
    table_name: str,
    toml_path: str | pathlib.Path | None = None,
) -> None:
    """Given a :class:`dict` containing the configuration for a *specific* top-level table,
    loaded from a toml file, validate the key names for that table,
    e.g. ``vak.train`` or ``vak.predict``.

    This function assumes ``table_config_dict`` comes from the entire ``config_dict``
    returned by :func:`vak.config.parse.from_toml_path`, accessed using the table name as a key,
    unlike :func:`are_keys_valid`. This function is used by the ``from_config_dict``
    classmethod of the top-level tables.
    """
    table_keys = set(table_config_dict.keys())
    valid_keys = set(VALID_KEYS[table_name])
    if not table_keys.issubset(valid_keys):
        invalid_keys = table_keys - valid_keys
        if toml_path:
            err_msg = (
                f"The following keys from '{table_name}' table in "
                f"the config file '{toml_path.name}' are not valid:\n{invalid_keys}"
            )
        else:
            err_msg = (
                f"The following keys from '{table_name}' table in "
                f"the toml config are not valid:\n{invalid_keys}"
            )
        raise ValueError(err_msg)
