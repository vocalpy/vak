"""Code that handles classes that represent the definition
of a neural network model; the abstraction of how models
are declared with code in vak."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Type, Union

import torch

REQUIRED_MODEL_DEFINITION_CLASS_VARS = (
    "network",
    "loss",
    "optimizer",
    "metrics",
    "default_config",
)

VALID_CONFIG_KEYS = REQUIRED_MODEL_DEFINITION_CLASS_VARS[
    :-1
]  # everything but 'default_config'


@dataclasses.dataclass
class ModelDefinition:
    """A class that represents the definition of a neural network model.

    A model definition is any class that has the following class variables:

    network: torch.nn.Module or dict
        Neural network.
        If a dict, should map string network names to torch.nn.Module classes.
    loss: torch.nn.Module, callable
        Either a built-in loss module, or a callable function that computes loss.
    optimizer: torch.optim.Optimizer
        Optimizer used to optimize neural network parameters during training.
    metrics: dict
        Metrics used to evaluate network. Should map string names of metric
        to callable classes that compute metric.
    default_config : dict
        That specifies default keyword arguments to use when instantiating any classes
        in ``network``, ``optimizer``, ``loss``, or ``metrics``.
        Used by ``vak.models.base.Model`` and its
        sub-classes that represent model families. E.g., those classes will do:
        ``network = self.definition.network(**self.definition.default_config['network'])``.

    Note it is **not** necessary to sub-class this class;
    it exists mainly for type-checking purposes.

    For more detail, see :func:`vak.models.decorator.model`
    and :class:`vak.models.ModelFactory`.
    """

    network: Union[torch.nn.Module, dict]
    loss: dict
    optimizer: torch.optim.Optimizer
    metrics: dict
    default_config: dict


# default that we set ``definition.default_config`` to,
# if definition does not have that class variable declared
DEFAULT_DEFAULT_CONFIG = {
    "network": {},
    "loss": {},
    "optimizer": {},
    "metrics": {},
}


def validate(definition: Type) -> Type:
    """Validate a model definition.

    A model definition is a class that has the following class variables:
        network: torch.nn.Module or dict
            Neural network.
            If a dict, should map string network names to torch.nn.Module classes.
        loss: torch.nn.Module, callable
            Either a built-in loss module, or a callable function that computes loss.
        optimizer: torch.optim.Optimizer
            Optimizer used to optimize neural network parameters during training.
        metrics: dict
            Metrics used to evaluate network. Should map string names of metric
            to callable classes that compute metric.
        default_config : dict
            That specifies default keyword arguments to use when instantiating any classes
            in ``network``, ``optimizer``, ``loss``, or ``metrics``.
            Used by ``vak.models.base.Model`` and its
            sub-classes that represent model families. E.g., those classes will do:
            ``network = self.definition.network(**self.definition.default_config['network'])``.
            If this class variable is not specified, it defaults to a ``dict``
            with the required keys, that map to empty ``dicts``.

    By providing this abstraction, ``vak`` commits in code
    to the idea that a neural network model consists of just
    the network function(s), the optimizer and the loss used
    to optimize the parameters of the network(s),
    as measured with the metrics.

    Parameters
    ----------
    definition : ModelDefinition
        A definition of a neural network model.
        A class having the class variables
        described above, with specific
        classes / callables / dicts assigned
        to those class variables.
        For an example, see
        ``vak.models.tweetynet.TweetyNet``.
        Does **not** need to be a sub-class of
        ``vak.models.definition.ModelDefinition``
        (that is used for type checking).

    Returns
    -------
    definition : type
        After validation, with ``default_config`` set to default
        if none was specified, as described above.

    Notes
    -----
    This function is used by the decorator
    ``vak.decorator.model``
    to validate a definition when
    converting it into a sub-class ofhttps://peps.python.org/pep-0416/
    ``vak.models.Model``.

    It's also used by :class:`vak.models.ModelFactory`,
    to validate a definition before building
    a new model instance from the definition.
    """
    # need to set this default first
    # so we don't throw error when checking class variables
    # if user did not specify ``default_config``
    if not hasattr(definition, "default_config"):
        definition.default_config = DEFAULT_DEFAULT_CONFIG
    else:
        # if they **did** specify ``default_config``,
        # make sure it's a dict
        if not isinstance(definition.default_config, dict):
            raise TypeError(
                "A model definition's ``default_config`` must be ``dict`` (or None)"
                f"but the type was: {type(definition.default_config)}"
            )

    # ---- check if any required class variables are missing
    definition_vars = {
        key: val
        for key, val in vars(definition).items()
        # keep class vars; throw out __module__, __doc__, etc.
        if not (key.startswith("__") and key.endswith("__"))
    }
    definition_class_var_names = list(definition_vars.keys())
    if not all(
        [
            expected_class_var_name in definition_class_var_names
            for expected_class_var_name in REQUIRED_MODEL_DEFINITION_CLASS_VARS
        ]
    ):
        missing_var_name = set(REQUIRED_MODEL_DEFINITION_CLASS_VARS) - set(
            definition_class_var_names
        )
        raise ValueError(
            f"Model definition is missing the following class variable(s): {missing_var_name}"
        )

    # ---- check if there are any extra class variables
    if any(
        [
            modeldef_var_name not in REQUIRED_MODEL_DEFINITION_CLASS_VARS
            for modeldef_var_name in definition_class_var_names
        ]
    ):
        extra_var_name = set(definition_class_var_names) - set(
            REQUIRED_MODEL_DEFINITION_CLASS_VARS
        )
        raise ValueError(
            f"Model definition has invalid class variable(s): {extra_var_name}."
            f"Valid class variables are: {REQUIRED_MODEL_DEFINITION_CLASS_VARS}"
        )

    # ---- now for each class variable check if they are the expected type.
    # either a torch.nn.Module or torch.optim.Optimizer subclass,
    # a dict mapping string names to torch.nn.Modules, or
    # a dict mapping string names to Callables.
    # Note that it's still hard to "unstringify" type annotations,
    # esp. in Python < 3.10, so
    # instead of getting it dynamically from __annotations__
    # we do validation "by hand" which is very verbose

    # ---- validate network
    network_obj = getattr(definition, "network")
    if inspect.isclass(network_obj):
        if not issubclass(network_obj, torch.nn.Module):
            raise TypeError(
                "A model definition's 'network' variable must be a subclass of torch.nn.Module "
                "or a dict mapping string names to torch.nn.Module subclasses, "
                f"but type was: {type(network_obj)}"
            )
    elif isinstance(network_obj, dict):
        for network_dict_key, network_dict_val in network_obj.items():
            if not isinstance(network_dict_key, str):
                raise TypeError(
                    "A model definition with a ``network`` variable that is a dict "
                    "should have keys that are strings, "
                    f"but the following key has type {type(network_dict_key)}: {network_dict_key}"
                )
            if not issubclass(network_dict_val, torch.nn.Module):
                raise TypeError(
                    "A model definition with a ``network`` variable that is a dict "
                    f"should have string keys mapping to values that are torch.nn.Module subclasses, "
                    f"but the following value has type {type(network_dict_val)}: {network_dict_val}"
                )
    else:
        raise TypeError(
            "A model definition's 'network' variable must be a subclass of torch.nn.Module "
            "or a dict mapping string names to torch.nn.Module subclasses, "
            f"but type was: {type(network_obj)}"
        )

    # ---- validate loss
    loss_obj = getattr(definition, "loss")
    # need complicated if-else here because issubclass throws an error if we don't pass it a class
    if inspect.isclass(loss_obj):
        if issubclass(loss_obj, torch.nn.Module):
            invalid_loss_obj_type = False
        else:
            invalid_loss_obj_type = True
    else:
        if inspect.isfunction(loss_obj):
            invalid_loss_obj_type = False
        else:
            invalid_loss_obj_type = True
    if invalid_loss_obj_type:
        raise TypeError(
            "A model definition's 'loss' variable must be a subclass of torch.nn.Module or a function, "
            f"but type was: {type(loss_obj)}"
        )

    # ---- validate optimizer
    optim_obj = getattr(definition, "optimizer")
    if not issubclass(optim_obj, torch.optim.Optimizer):
        raise TypeError(
            "A model definition's 'optimizer' variable must be a subclass of torch.optim.Optimizer, "
            f"but type was: {type(optim_obj)}"
        )

    # ---- validate metrics
    metrics_obj = getattr(definition, "metrics")
    if not isinstance(metrics_obj, dict):
        raise TypeError(
            "A model definition's 'metrics' variable must be a dict mapping string names to callables, "
            f"but was type: {type(metrics_obj)}"
        )
    for metrics_dict_key, metrics_dict_val in metrics_obj.items():
        if not isinstance(metrics_dict_key, str):
            raise TypeError(
                f"A model definition's 'metrics' variable must be a dict mapping string names to callables, "
                f"but the following key has type {type(metrics_dict_key)}: {metrics_dict_key}"
            )
        if not (
            inspect.isclass(metrics_dict_val) and callable(metrics_dict_val)
        ):
            raise TypeError(
                "A model definition's 'metrics' variable must be a dict mapping "
                "string names to classes that define __call__ methods, "
                f"but the key '{metrics_dict_key}' maps to a value with type {type(metrics_dict_val)}, "
                f"not recognized as callable."
            )

    # ---- validate default config
    default_config = getattr(definition, "default_config")

    if not all(
        [
            config_key in VALID_CONFIG_KEYS
            for config_key in default_config.keys()
        ]
    ):
        invalid_keys = [
            config_key
            for config_key in default_config.keys()
            if config_key not in VALID_CONFIG_KEYS
        ]
        raise ValueError(
            f"Invalid keys in default_config: {invalid_keys}."
            f"Valid keys are: {VALID_CONFIG_KEYS}"
        )

    # -------- validate 'network' config
    network_config = default_config.get("network")
    if network_config is None:
        if inspect.isclass(definition.network):
            # calling 'if issubclass(definition.network, torch.nn.Module)'
            # would raise an error when definition.network is a dict
            definition.default_config["network"] = {}
        elif isinstance(definition.network, dict):
            definition.default_config["network"] = {
                network_name: {} for network_name in definition.network.keys()
            }
    elif len(network_config) > 0:
        if inspect.isclass(definition.network):
            # calling 'if issubclass(definition.network, torch.nn.Module)'
            # would raise an error when definition.network is a dict
            network_init_params = list(
                inspect.signature(
                    definition.network.__init__
                ).parameters.keys()
            )
            if any(
                [
                    network_kwarg not in network_init_params
                    for network_kwarg in network_config.keys()
                ]
            ):
                invalid_keys = set(network_config.keys()) - set(
                    network_init_params
                )
                raise ValueError(
                    f"The following keyword arguments specified in the ``default_config`` "
                    f"for ``network`` are invalid: {invalid_keys}."
                    f"Valid arguments are: {network_init_params}"
                )

        elif isinstance(definition.network, dict):
            if any(
                [
                    network_name not in definition.network.keys()
                    for network_name in network_config.keys()
                ]
            ):
                invalid_network_names = [
                    network_name
                    for network_name in network_config.keys()
                    if network_name not in definition.network.keys()
                ]

                raise ValueError(
                    "When model definition's ``network`` is a ``dict`` mapping string names to ``torch.nn.Module``s,"
                    "the definition's ``default_config`` should have only those string names as keys."
                    f"The following keys in the default_config for network are invalid: {invalid_network_names}."
                    f"Valid keys are these network names: {definition.network.keys()}"
                    "Please rewrite ``default_config`` so keys of ``default_config['network']`` "
                    "are only those string names, "
                    "and the corresponding values for those keys are keyword arguments for the networks."
                )
            for network_name, network_kwargs in network_config.items():
                network_init_params = list(
                    inspect.signature(
                        definition.network[network_name].__init__
                    ).parameters.keys()
                )
                if any(
                    [
                        network_kwarg not in network_init_params
                        for network_kwarg in network_kwargs.keys()
                    ]
                ):
                    invalid_keys = set(network_config.keys()) - set(
                        network_init_params
                    )
                    raise ValueError(
                        f"The following keyword arguments specified in the ``default_config`` "
                        f"for ``network`` are invalid: {invalid_keys}."
                        f"Valid arguments are: {network_init_params}"
                    )

    # -------- validate 'loss' config
    loss_config = default_config.get("loss")
    if loss_config is None:
        definition.default_config["loss"] = {}
    elif len(loss_config) > 0:
        if inspect.isfunction(definition.loss):
            raise ValueError(
                "Model definition's default_config specifies keyword arguments for loss, "
                "but loss is a function, not a class. Please only specify keyword arguments for classes."
            )
        loss_init_params = list(
            inspect.signature(definition.loss.__init__).parameters.keys()
        )
        if any(
            [
                loss_kwarg not in loss_init_params
                for loss_kwarg in loss_config.keys()
            ]
        ):
            invalid_loss_kwargs = set(loss_config.keys()) - set(
                loss_init_params
            )
            raise ValueError(
                f"The following keyword arguments specified in the ``default_config`` "
                f"for ``loss`` are invalid: {invalid_loss_kwargs}."
                f"Valid arguments are: {loss_init_params}"
            )

    # -------- validate 'optimizer' config
    optimizer_config = default_config.get("optimizer")
    if optimizer_config is None:
        definition.default_config["optimizer"] = {}
    elif len(optimizer_config) > 0:
        optimizer_init_params = list(
            inspect.signature(definition.optimizer.__init__).parameters.keys()
        )
        if any(
            [
                optimizer_kwarg not in optimizer_init_params
                for optimizer_kwarg in optimizer_config.keys()
            ]
        ):
            invalid_optimizer_kwargs = set(optimizer_config.keys()) - set(
                optimizer_init_params
            )
            raise ValueError(
                f"The following keyword arguments specified in the ``default_config`` "
                f"for ``optimizer`` are invalid: {invalid_optimizer_kwargs}."
                f"Valid arguments are: {optimizer_init_params}"
            )

    # -------- validate 'metrics' config
    metrics_config = default_config.get("metrics")
    if metrics_config is None:
        definition.default_config["metrics"] = {}
    elif len(metrics_config) > 0:
        if any(
            [
                metric_name not in definition.metrics
                for metric_name in metrics_config.keys()
            ]
        ):
            invalid_metric_names = set(metrics_config.keys()) - set(
                definition.metrics.keys()
            )
            raise ValueError(
                f"The following metric names specified in the ``default_config`` "
                f"for ``metrics`` are invalid: {invalid_metric_names}."
                f"Valid metric names are: {definition.metrics.keys()}"
            )
        for metric_name, metric_class_config in metrics_config.items():
            metric_class_init_params = list(
                inspect.signature(
                    definition.metrics[metric_name].__init__
                ).parameters.keys()
            )
            if any(
                [
                    metric_class_kwarg not in metric_class_init_params
                    for metric_class_kwarg in metric_class_config.keys()
                ]
            ):
                invalid_metric_class_kwargs = set(
                    metric_class_config.keys()
                ) - set(metric_class_init_params)
                raise ValueError(
                    f"The following keyword arguments specified in the ``default_config`` "
                    f"for 'metrics' class {definition.metrics[metric_name]} are invalid: "
                    f"{invalid_metric_class_kwargs}."
                    f"Valid arguments are: {metric_class_init_params}"
                )

    return definition
