"""Class representing the model table of a toml configuration file."""
from __future__ import annotations

import pathlib

from attrs import define, field
from attrs.validators import instance_of

from .. import models


MODEL_TABLES = [
    "network",
    "optimizer",
    "loss",
    "metrics",
]


@define
class ModelConfig:
    """Class representing the model table of a toml configuration file.

    Attributes
    ----------
    name : str
    network : dict
        Keyword arguments for the network class,
        or a :class:`dict` of ``dict``s mapping
        network names to keyword arguments.
    optimizer: dict
        Keyword arguments for the optimizer class.
    loss : dict
        Keyword arguments for the class representing the loss function.
    metrics: dict
        A :class:`dict` of ``dict``s mapping
        metric names to keyword arguments.
    """
    name: str
    network: dict = field(validator=instance_of(dict))
    optimizer: dict = field(validator=instance_of(dict))
    loss: dict = field(validator=instance_of(dict))
    metrics: dict = field(validator=instance_of(dict))

    @classmethod
    def from_config_dict(cls, config_dict: dict):
        """Return :class:`ModelConfig` instance from a :class:`dict`.

        The :class:`dict` passed in should be the one found
        by loading a valid configuration toml file with
        :func:`vak.config.parse.from_toml_path`,
        and then using a top-level table key,
        followed by key ``'model'``.
        E.g., ``config_dict['train']['model']` or
        ``config_dict['predict']['model']``.

        Examples
        --------
        config_dict = vak.config.parse.from_toml_path(toml_path)
        model_config = vak.config.Model.from_config_dict(config_dict['train'])
        """
        model_name = list(config_dict.keys())
        if len(model_name) == 0:
            raise ValueError(
                "Did not find a single key in `config_dict` corresponding to model name. "
                f"Instead found no keys. Config dict:\n{config_dict}\n"
                "A configuration file should specify a single model per top-level table."
            )
        if len(model_name) > 1:
            raise ValueError(
                "Did not find a single key in `config_dict` corresponding to model name. "
                f"Instead found multiple keys: {model_name}.\nConfig dict:\n{config_dict}.\n"
                "A configuration file should specify a single model per top-level table."
            )
        model_name = model_name[0]
        MODEL_NAMES = list(models.registry.MODEL_NAMES)
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f"Model name not found in registry: {model_name}\n"
                f"Model names in registry:\n{MODEL_NAMES}"
            )
        model_config = config_dict[model_name]
        if not all(
            key in MODEL_TABLES for key in model_config.keys()
        ):
            invalid_keys = (
                key for key in model_config.keys() if key not in MODEL_TABLES
            )
            raise ValueError(
                f"The following sub-tables in the model config are not valid: {invalid_keys}\n"
                f"Valid sub-table names are: {MODEL_TABLES}"
            )
        # for any tables not specified, default to empty dict so we can still use ``**`` operator on it
        for model_table in MODEL_TABLES:
            if model_table not in config_dict:
                model_config[model_table] = {}
        return cls(
            name=model_name,
            **model_config
        )


def config_from_toml_dict(toml_dict: dict, table: str, model_name: str) -> dict:
    """Get configuration for a model from a .toml configuration file
    loaded into a ``dict``.

    Parameters
    ----------
    toml_dict : dict
        Configuration from a .toml file, loaded into a dictionary.
    table : str
        Name of top-level table to get model config from.
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
    from . import validators  # avoid circular import
    validators.are_tables_valid(toml_dict)

    try:
        model_config = toml_dict[table][model_name]
    except KeyError as e:
        raise ValueError(
            f"A config section specifies the model name '{model_name}', "
            f"but there is no section named '{model_name}' in the '{table}' table of the config."
        ) from e

    # check if config declares parameters for required attributes;
    # if not, just put an empty dict that will get passed as the "kwargs"
    for attr in MODEL_TABLES:
        if attr not in model_config:
            model_config[attr] = {}

    return model_config


def config_from_toml_path(
    toml_path: str | pathlib.Path, table: str, model_name: str
) -> dict:
    """Get configuration for a model from a .toml configuration file,
    given the path to the file.

    Parameters
    ----------
    toml_path : str, Path
        to configuration file in .toml format
    table : str
        Name of top-level table to get model config from.
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
    from . import parse  # avoid circular import

    toml_dict = parse._load_toml_from_path(toml_path)
    return config_from_toml_dict(toml_dict, table, model_name)
