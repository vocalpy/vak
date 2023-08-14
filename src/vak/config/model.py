from __future__ import annotations

import pathlib

import toml

from .. import models

MODEL_TABLES = [
    "network",
    "optimizer",
    "loss",
    "metrics",
]


def config_from_toml_dict(toml_dict: dict, model_name: str) -> dict:
    """Get configuration for a model from a .toml configuration file
    loaded into a ``dict``.

    Parameters
    ----------
    toml_dict : dict
        Configuration from a .toml file, loaded into a dictionary.
    model_name : str
        Name of a model, specified as the ``model`` option in a table
        (such as TRAIN or PREDICT),
        that should have its own corresponding table
        specifying its configuration: hyperparameters such as learning rate, etc.

    Returns
    -------
    model_config : dict
        Model configuration in a ``dict``,
        as loaded from a .toml file,
        and used by the model method ``from_config``.
    """
    if model_name not in models.registry.MODEL_NAMES:
        raise ValueError(
            f"Invalid model name: {model_name}.\nValid model names are: {models.registry.MODEL_NAMES}"
        )

    try:
        model_config = toml_dict[model_name]
    except KeyError as e:
        raise ValueError(
            f"A config section specifies the model name '{model_name}', "
            f"but there is no section named '{model_name}' in the config."
        ) from e

    # check if config declares parameters for required attributes;
    # if not, just put an empty dict that will get passed as the "kwargs"
    for attr in MODEL_TABLES:
        if attr not in model_config:
            model_config[attr] = {}

    return model_config


def config_from_toml_path(
    toml_path: str | pathlib.Path, model_name: str
) -> dict:
    """Get configuration for a model from a .toml configuration file,
    given the path to the file.

    Parameters
    ----------
    toml_path : str, Path
        to configuration file in .toml format
     model_name : str
        of str, i.e. names of models specified by a section
        (such as TRAIN or PREDICT) that should each have corresponding sections
        specifying their configuration: hyperparameters such as learning rate, etc.

    Returns
    -------
    model_config : dict
        Model configuration in a ``dict``,
        as loaded from a .toml file,
        and used by the model method ``from_config``.
    """
    toml_path = pathlib.Path(toml_path)
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"File not found, or not recognized as a file: {toml_path}"
        )

    with toml_path.open("r") as fp:
        config_dict = toml.load(fp)
    return config_from_toml_dict(config_dict, model_name)
