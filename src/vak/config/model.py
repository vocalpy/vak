"""Class representing the model table of a toml configuration file."""

from __future__ import annotations

from attrs import asdict, define, field
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

        # NOTE: we are getting model_config here
        model_config = config_dict[model_name]
        if not all(key in MODEL_TABLES for key in model_config.keys()):
            invalid_keys = (
                key for key in model_config.keys() if key not in MODEL_TABLES
            )
            raise ValueError(
                f"The following sub-tables in the model config are not valid: {invalid_keys}\n"
                f"Valid sub-table names are: {MODEL_TABLES}"
            )
        # for any tables not specified, default to empty dict so we can still use ``**`` operator on it
        for model_table in MODEL_TABLES:
            if model_table not in model_config:
                model_config[model_table] = {}
        return cls(name=model_name, **model_config)

    def asdict(self):
        """Convert this :class:`ModelConfig` instance
        to a :class:`dict` that can be passed
        into functions that take a ``model_config`` argument,
        like :func:`vak.train` and :func:`vak.predict`.
        """
        return asdict(self)
