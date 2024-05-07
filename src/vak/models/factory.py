"""Class that represent a model builit into ``vak``."""

from __future__ import annotations

import inspect
from typing import Callable, Type

import lightning
import torch

from .decorator import ModelDefinitionValidationError
from .definition import validate as validate_definition


class ModelFactory:
    """Class that represent a model builit into ``vak``.

    Attributes
    ----------
    definition: vak.models.definition.ModelDefinition
    family: lighting.pytorch.LightningModule

    Notes
    -----
    This class is used by the :func:`vak.models.decorator.model`
    decorator to make a new class representing a model
    from a model definition.
    As such, this class is not meant to be used directly.
    See the docstring of :func:`vak.models.decorator.model`
    for more detail.
    """

    def __init__(
        self,
        definition: Type,
        family: lightning.pytorch.LightningModule,
    ) -> None:
        if not issubclass(family, lightning.pytorch.LightningModule):
            raise TypeError(
                "The ``family`` argument to the ``vak.models.model`` decorator"
                "should be a subclass of ``lightning.pytorch.LightningModule``,"
                f"but the type was: {type(family)}, "
                "which was not recognized as a subclass "
                "of ``lightning.pytorch.LightningModule``."
            )

        try:
            validate_definition(definition)
        except ValueError as err:
            raise ModelDefinitionValidationError(
                f"Validation failed for the following model definition:\n{definition}"
            ) from err
        except TypeError as err:
            raise ModelDefinitionValidationError(
                f"Validation failed for the following model definition:\n{definition}"
            ) from err

        self.definition = definition
        self.family = family

    def attributes_from_config(self, config: dict):
        """Get attributes for an instance of a model,
        given a configuration.

        Given a :class:`dict`, ``config``, return instances
        of `network`, `optimizer`, `loss`, and `metrics`.

        Parameters
        ----------
        config : dict
            A :class:`dict` obtained by calling
            :meth:`vak.config.ModelConfig.to_dict()`.

        Returns
        -------
        network : torch.nn.Module, dict
            An instance of a ``torch.nn.Module``
            that implements a neural network,
            or a ``dict`` that maps human-readable string names
            to a set of such instances.
        loss : torch.nn.Module, callable
            An instance of a ``torch.nn.Module``
            that implements a loss function.
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
            "network", self.definition.default_config["network"]
        )
        if inspect.isclass(self.definition.network):
            network = self.definition.network(**network_kwargs)
        elif isinstance(self.definition.network, dict):
            network = {}
            for net_name, net_class in self.definition.network.items():
                net_class_kwargs = network_kwargs.get(net_name, {})
                network[net_name] = net_class(**net_class_kwargs)

        if isinstance(self.definition.network, dict):
            params = [
                param
                for net_name, net_instance in network.items()
                for param in net_instance.parameters()
            ]
        else:
            params = network.parameters()

        optimizer_kwargs = config.get(
            "optimizer", self.definition.default_config["optimizer"]
        )
        optimizer = self.definition.optimizer(
            params=params, **optimizer_kwargs
        )

        if inspect.isclass(self.definition.loss):
            loss_kwargs = config.get(
                "loss", self.definition.default_config["loss"]
            )
            loss = self.definition.loss(**loss_kwargs)
        else:
            loss = self.definition.loss

        metrics_config = config.get(
            "metrics", self.definition.default_config["metrics"]
        )
        metrics = {}
        for metric_name, metric_class in self.definition.metrics.items():
            metrics_class_kwargs = metrics_config.get(metric_name, {})
            metrics[metric_name] = metric_class(**metrics_class_kwargs)

        return network, loss, optimizer, metrics

    def validate_init(
        self,
        network: torch.nn.Module | dict | None = None,
        loss: torch.nn.Module | Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict | None = None,
    ):
        """Validate arguments to ``vak.models.base.Model.init``.

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
            if inspect.isclass(self.definition.network):
                if not isinstance(network, self.definition.network):
                    raise TypeError(
                        f"``network`` should be an instance of {self.definition.network}"
                        f"but was of type {type(network)}"
                    )

            elif isinstance(self.definition.network, dict):
                if not isinstance(network, dict):
                    raise TypeError(
                        "Expected ``network`` to be a ``dict`` mapping network names "
                        f"to ``torch.nn.Module`` instances, but type was {type(network)}"
                    )
                expected_network_dict_keys = list(
                    self.definition.network.keys()
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
                        network_instance, self.definition.network[network_name]
                    ):
                        raise TypeError(
                            f"Network with name '{network_name}' in ``network`` dict "
                            f"should be an instance of {self.definition.network[network_name]}"
                            f"but was of type {type(network)}"
                        )
            else:
                raise TypeError(
                    f"Invalid type for ``network``: {type(network)}"
                )

        if loss:
            if issubclass(self.definition.loss, torch.nn.Module):
                if not isinstance(loss, self.definition.loss):
                    raise TypeError(
                        f"``loss`` should be an instance of {self.definition.loss}"
                        f"but was of type {type(loss)}"
                    )
            elif callable(self.definition.loss):
                if loss is not self.definition.loss:
                    raise ValueError(
                        f"``loss`` should be the following callable (probably a function): {self.definition.loss}"
                    )
            else:
                raise TypeError(f"Invalid type for ``loss``: {type(loss)}")

        if optimizer:
            if not isinstance(optimizer, self.definition.optimizer):
                raise TypeError(
                    f"``optimizer`` should be an instance of {self.definition.optimizer}"
                    f"but was of type {type(optimizer)}"
                )

        if metrics:
            if not isinstance(metrics, dict):
                raise TypeError(
                    "``metrics`` should be a ``dict`` mapping string metric names "
                    f"to callable metrics, but type of ``metrics`` was {type(metrics)}"
                )
            for metric_name, metric_callable in metrics.items():
                if metric_name not in self.definition.metrics:
                    raise ValueError(
                        f"``metrics`` has name '{metric_name}' but that name "
                        f"is not in the model definition. "
                        f"Valid metric names are: {', '.join(list(self.definition.metrics.keys()))}"
                    )

                if not isinstance(
                    metric_callable, self.definition.metrics[metric_name]
                ):
                    raise TypeError(
                        f"metric '{metric_name}' should be an instance of {self.definition.metrics[metric_name]}"
                        f"but was of type {type(metric_callable)}"
                    )

    def validate_instances_or_get_default(
        self,
        network: torch.nn.Module | dict | None = None,
        loss: torch.nn.Module | Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict | None = None,
    ):
        """Validate instances of model attributes, using its definition,
        or if no instance is passed in for an attribute,
        make an instance using the default config.

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

        if loss is None:
            if inspect.isclass(self.definition.loss):
                loss_kwargs = self.definition.default_config.get("loss")
                loss = self.definition.loss(**loss_kwargs)
            elif inspect.isfunction(self.definition.loss):
                loss = self.definition.loss

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

        if metrics is None:
            metric_kwargs = self.definition.default_config.get("metrics")
            metrics = {}
            for metric_name, metric_class in self.definition.metrics.items():
                metric_class_kwargs = metric_kwargs.get(metric_name, {})
                metrics[metric_name] = metric_class(**metric_class_kwargs)
        return network, loss, optimizer, metrics

    def from_config(self, config: dict, **kwargs):
        """Return a a new instance of a model, given a config :class:`dict`.

        Parameters
        ----------
        config : dict
            The dict obtained by by calling :meth:`vak.config.ModelConfig.asdict`.

        Returns
        -------
        model : lightning.LightningModule
            An instance of the model :attr:`~ModelFactory.family`
            with attributes specified by :attr:`~ModelFactory.definition`,
            that are initialized using parameters from ``config``.
        """
        network, loss, optimizer, metrics = self.attributes_from_config(config)
        network, loss, optimizer, metrics = (
            self.validate_instances_or_get_default(
                network,
                loss,
                optimizer,
                metrics,
            )
        )
        return self.family(
            network=network,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            **kwargs,
        )

    def from_instances(
        self,
        network: torch.nn.Module | dict | None = None,
        loss: torch.nn.Module | Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict | None = None,
        **kwargs,
    ):
        """

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
        network, loss, optimizer, metrics = (
            self.validate_instances_or_get_default(
                network,
                loss,
                optimizer,
                metrics,
            )
        )
        return self.family(
            network=network,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            **kwargs,
        )
