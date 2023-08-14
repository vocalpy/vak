"""Base class for a model in ``vak``,
that other families of models should subclass.
"""
from __future__ import annotations

import inspect
from typing import Callable, ClassVar

import pytorch_lightning as lightning
import torch

from .definition import ModelDefinition
from .definition import validate as validate_definition


class Model(lightning.LightningModule):
    """Base class for a model in ``vak``,
    that other families of models should subclass.

    This class provides methods for working with
    neural network models, e.g. training the model
    and generating productions,
    and it also converts a
    model definition into a model instance.

    It provides the methods for working with neural
    network models by subclassing
    ``lighting.LightningModule``, and it handles
    converting a model definition into a model instance
    inside its ``__init__`` method.
    Model definitions are declared programmatically
    using a ``vak.model.ModelDefinition``;
    see the documentation on that class for more detail.
    """

    definition: ClassVar[ModelDefinition]

    def __init__(
        self,
        network: torch.nn.Module | dict | None = None,
        loss: torch.nn.Module | Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict | None = None,
    ):
        """Initializes an instance of a model, using its definition.

        Takes in instances of the attributes defined by the class variable
        ``self.definition``: ``network``, ``loss``, ``optimizer``, and ``metrics``.
        If any of those arguments are ``None``, then ``__init__``
        instantiates the corresponding attribute with its defaults.
        If any of those arguments are not an instance of the type
        defined by ``self.definition``, then a TypeError is raised.

        Parameters
        ----------
        network : torch.nn.Module, dict
            An instance of a ``torch.nn.Module``
            that implements a neural network,
            or a ``dict`` that maps human-readable string names
            to a set of such instances.
        loss : torch.nn.Module, callable
            An instance of a ``torch.nn.Module``
            that implements a loss function,
            or a callable Python function that
            computes a scalar loss.
        optimizer : torch.optim.Optimizer
            An instance of a ``torch.optim.Optimizer`` class
            used with ``loss`` to optimize
            the parameters of ``network``.
        metrics : dict
            A ``dict`` that maps human-readable string names
            to ``Callable`` functions, used to measure
            performance of the model.
        """
        from .decorator import ModelDefinitionValidationError

        super().__init__()

        # check that we are a sub-class of some other class with required class variables
        if not hasattr(self, "definition"):
            raise ValueError(
                "This model does not have a definition."
                "Define a model by wrapping a class with the required class variables with "
                "a ``vak.models`` decorator, e.g. ``vak.models.windowed_frame_classification_model``"
            )

        try:
            validate_definition(self.definition)
        except ModelDefinitionValidationError as err:
            raise ValueError(
                "Creating model instance failed because model definition is invalid."
            ) from err

        # ---- validate any instances that user passed in
        self.validate_init(network, loss, optimizer, metrics)

        if network is None:
            net_kwargs = self.definition.default_config.get("network")
            if isinstance(self.definition.network, dict):
                network = {
                    network_name: network_class(**net_kwargs[network_name])
                    for network_name, network_class in self.definition.network.items()
                }
            else:
                network = self.definition.network(**net_kwargs)
        self.network = network

        if loss is None:
            if inspect.isclass(self.definition.loss):
                loss_kwargs = self.definition.default_config.get("loss")
                loss = self.definition.loss(**loss_kwargs)
            elif inspect.isfunction(self.definition.loss):
                loss = self.definition.loss
        self.loss = loss

        if optimizer is None:
            optimizer_kwargs = self.definition.default_config.get("optimizer")
            if isinstance(network, dict):
                params = [
                    param
                    for net_name, net_instance in network.items()
                    for param in net_instance.parameters()
                ]
            else:
                params = network.parameters()
            optimizer = self.definition.optimizer(
                params=params, **optimizer_kwargs
            )
        self.optimizer = optimizer

        if metrics is None:
            metric_kwargs = self.definition.default_config.get("metrics")
            metrics = {}
            for metric_name, metric_class in self.definition.metrics.items():
                metric_class_kwargs = metric_kwargs.get(metric_name, {})
                metrics[metric_name] = metric_class(**metric_class_kwargs)
        self.metrics = metrics

    @classmethod
    def validate_init(
        cls,
        network: torch.nn.Module | dict | None = None,
        loss: torch.nn.Module | Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict | None = None,
    ):
        """Validate arguments to ``vak.models.base.Model.__init__``.

        Parameters
        ----------
        network : torch.nn.Module, dict
            An instance of a ``torch.nn.Module``
            that implements a neural network,
            or a ``dict`` where each key is a string
             that maps a human-readable name
            to a ``torch.nn.Module`` instance.
        loss : torch.nn.Module, callable
            An instance of a ``torch.nn.Module``
            that implements a loss function,
            or a callable Python function that
            computes a scalar loss.
        optimizer : torch.optim.Optimizer
            An instance of a ``torch.optim.Optimizer`` class
            used with ``loss`` to optimize
            the parameters of ``network``.
        metrics : dict
            A ``dict`` that maps human-readable string names
            to ``Callable`` functions, used to measure
            performance of the model.

        Returns
        -------
        None

        This method does not return values;
        it just raises an error if any value is invalid.
        """
        if network:
            if inspect.isclass(cls.definition.network):
                if not isinstance(network, cls.definition.network):
                    raise TypeError(
                        f"``network`` should be an instance of {cls.definition.network}"
                        f"but was of type {type(network)}"
                    )

            elif isinstance(cls.definition.network, dict):
                if not isinstance(network, dict):
                    raise TypeError(
                        "Expected ``network`` to be a ``dict`` mapping network names "
                        f"to ``torch.nn.Module`` instances, but type was {type(network)}"
                    )
                expected_network_dict_keys = list(
                    cls.definition.network.keys()
                )
                network_dict_keys = list(network.keys())
                if not all(
                    [
                        expected_network_dict_key in network_dict_keys
                        for expected_network_dict_key in expected_network_dict_keys
                    ]
                ):
                    missing_keys = set(expected_network_dict_keys) - set(
                        network_dict_keys
                    )
                    raise ValueError(
                        f"The following keys were missing from the ``network`` dict: {missing_keys}"
                    )
                if any(
                    [
                        network_dict_key not in expected_network_dict_keys
                        for network_dict_key in network_dict_keys
                    ]
                ):
                    extra_keys = set(network_dict_keys) - set(
                        expected_network_dict_keys
                    )
                    raise ValueError(
                        f"The following keys in the ``network`` dict are not valid: {extra_keys}."
                        f"Valid keys are: {expected_network_dict_keys}"
                    )

                for network_name, network_instance in network.items():
                    if not isinstance(
                        network_instance, cls.definition.network[network_name]
                    ):
                        raise TypeError(
                            f"Network with name '{network_name}' in ``network`` dict "
                            f"should be an instance of {cls.definition.network[network_name]}"
                            f"but was of type {type(network)}"
                        )
            else:
                raise TypeError(
                    f"Invalid type for ``network``: {type(network)}"
                )

        if loss:
            if issubclass(cls.definition.loss, torch.nn.Module):
                if not isinstance(loss, cls.definition.loss):
                    raise TypeError(
                        f"``loss`` should be an instance of {cls.definition.loss}"
                        f"but was of type {type(loss)}"
                    )
            elif callable(cls.definition.loss):
                if loss is not cls.definition.loss:
                    raise ValueError(
                        f"``loss`` should be the following callable (probably a function): {cls.definition.loss}"
                    )
            else:
                raise TypeError(f"Invalid type for ``loss``: {type(loss)}")

        if optimizer:
            if not isinstance(optimizer, cls.definition.optimizer):
                raise TypeError(
                    f"``optimizer`` should be an instance of {cls.definition.optimizer}"
                    f"but was of type {type(optimizer)}"
                )

        if metrics:
            if not isinstance(metrics, dict):
                raise TypeError(
                    "``metrics`` should be a ``dict`` mapping string metric names "
                    f"to callable metrics, but type of ``metrics`` was {type(metrics)}"
                )
            for metric_name, metric_callable in metrics.items():
                if metric_name not in cls.definition.metrics:
                    raise ValueError(
                        f"``metrics`` has name '{metric_name}' but that name "
                        f"is not in the model definition. "
                        f"Valid metric names are: {', '.join(list(cls.definition.metrics.keys()))}"
                    )

                if not isinstance(
                    metric_callable, cls.definition.metrics[metric_name]
                ):
                    raise TypeError(
                        f"metric '{metric_name}' should be an instance of {cls.definition.metrics[metric_name]}"
                        f"but was of type {type(metric_callable)}"
                    )

    def load_state_dict_from_path(self, ckpt_path):
        """Loads a model from the path to a saved checkpoint.

        Loads the checkpoint and then calls
        ``self.load_state_dict`` with the ``state_dict``
        in that chekcpoint.

        This method allows loading a state dict into an instance.
        It's necessary because `lightning.LightningModule.load`` is a
        ``classmethod``, so calling that method will trigger
         ``LightningModule.__init__`` instead of running
        ``vak.models.Model.__init__``.

        Parameters
        ----------
        ckpt_path : str, pathlib.Path
            Path to a checkpoint saved by a model in ``vak``.
            This checkpoint has the same key-value pairs as
            any other checkpoint saved by a
            ``lightning.LightningModule``.

        Returns
        -------
        None

        This method modifies the model state by loading the ``state_dict``;
        it does not return anything.
        """
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["state_dict"])

    @classmethod
    def attributes_from_config(cls, config: dict):
        """Get attributes for an instance of a model,
        given a configuration.

        Given a ``dict``, ``config``, return instances of
        class variables

        Parameters
        ----------
        config : dict
            Returned by calling ``vak.config.models.map_from_path``
            or ``vak.config.models.map_from_config_dict``.

        Returns
        -------
        network : torch.nn.Module, dict
            An instance of a ``torch.nn.Module``
            that implements a neural network,
            or a ``dict`` that maps human-readable string names
            to a set of such instances.
        loss : torch.nn.Module, callable
            An instance of a ``torch.nn.Module``
            that implements a loss function,
            or a callable Python function that
            computes a scalar loss.
        optimizer : torch.optim.Optimizer
            An instance of a ``torch.optim.Optimizer`` class
            used with ``loss`` to optimize
            the parameters of ``network``.
        metrics : dict
            A ``dict`` that maps human-readable string names
            to ``Callable`` functions, used to measure
            performance of the model.
        """
        network_kwargs = config.get(
            "network", cls.definition.default_config["network"]
        )
        if inspect.isclass(cls.definition.network):
            network = cls.definition.network(**network_kwargs)
        elif isinstance(cls.definition.network, dict):
            network = {}
            for net_name, net_class in cls.definition.network.items():
                net_class_kwargs = network_kwargs.get(net_name, {})
                network[net_name] = net_class(**net_class_kwargs)

        if isinstance(cls.definition.network, dict):
            params = [
                param
                for net_name, net_instance in network.items()
                for param in net_instance.parameters()
            ]
        else:
            params = network.parameters()

        optimizer_kwargs = config.get(
            "optimizer", cls.definition.default_config["optimizer"]
        )
        optimizer = cls.definition.optimizer(params=params, **optimizer_kwargs)

        if inspect.isclass(cls.definition.loss):
            loss_kwargs = config.get(
                "loss", cls.definition.default_config["loss"]
            )
            loss = cls.definition.loss(**loss_kwargs)
        else:
            loss = cls.definition.loss

        metrics_config = config.get(
            "metrics", cls.definition.default_config["metrics"]
        )
        metrics = {}
        for metric_name, metric_class in cls.definition.metrics.items():
            metrics_class_kwargs = metrics_config.get(metric_name, {})
            metrics[metric_name] = metric_class(**metrics_class_kwargs)

        return network, loss, optimizer, metrics

    @classmethod
    def from_config(cls, config: dict):
        """Return an initialized model instance from a config ``dict``

        Parameters
        ----------
        config : dict
            Returned by calling ``vak.config.models.map_from_path``
            or ``vak.config.models.map_from_config_dict``.

        Returns
        -------
        cls : vak.models.base.Model
            An instance of the model with its attributes
            initialized using parameters from ``config``.
        """
        network, loss, optimizer, metrics = cls.attributes_from_config(config)
        return cls(
            network=network, loss=loss, optimizer=optimizer, metrics=metrics
        )
