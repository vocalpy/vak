"""Registry for models.

Makes it possible to register a model declared outside of ``vak``
with a decorator, so that the model can be used at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type

import lightning

if TYPE_CHECKING:
    from .factory import ModelFactory

MODEL_FAMILY_REGISTRY = {}


def model_family(family_class: Type) -> None:
    """Decorator that adds a :class:`lightning.LightningModule` class to the registry of model families."""
    if not issubclass(family_class, lightning.LightningModule):
        raise TypeError(
            "The ``family_class`` provided to the `vak.models.model_family` decorator"
            "must be a subclass of `lightning.LightningModule`, "
            f"but the class specified is not: {family_class}. "
            f"Subclasses of `lightning.LightningModule` are: {lightning.LightningModule.__subclasses__()}"
        )

    model_family_name = family_class.__name__
    if model_family_name in MODEL_FAMILY_REGISTRY:
        raise ValueError(
            f"Attempted to register a model family with the name '{model_family_name}', "
            f"but this name is already in the registry:\n{MODEL_FAMILY_REGISTRY}"
        )

    MODEL_FAMILY_REGISTRY[model_family_name] = family_class
    # need to return class after we register it or we replace it with None
    # when this function is used as a decorator
    return family_class


MODEL_REGISTRY = {}


def register_model(model: ModelFactory) -> ModelFactory:
    """Function that registers a model in the model registry.

    This function is called by :func:`vak.models.decorator.model`,
    that creates an instance of a :class:`vak.models.ModelFactory`,
    given a :class:`vak.models.definition.ModelDefinition`
    and a :class:`lightning.LightningModule` class that has been
    registered as a model family with :func:`model_family`.

    So you will not usually need to use this function directly,
    and should prefer to use :func:`vak.models.decorator.model` instead.
    """
    model_family_classes = list(MODEL_FAMILY_REGISTRY.values())
    model_family = model.family
    if model_family not in model_family_classes:
        raise TypeError(
            "The family of `model` passed to the `register_model` decorator "
            f"is not recognized as a model family. Class was '{model}' and "
            f"its family is '{model_family}'. "
            f"Please specify a valid model family. "
            f"Valid model family classes are: {model_family_classes}"
        )

    model_name = model.__name__
    if model_name in MODEL_REGISTRY:
        raise ValueError(
            f"Attempted to register a model family with the name '{model_name}', "
            f"but this name is already in the registry.\n"
        )

    MODEL_REGISTRY[model_name] = model
    # need to return class after we register it,
    # or we would replace it with None when this function is used as a decorator
    return model


def __getattr__(name: str) -> Any:
    """Module-level __getattr__ function that we use to dynamically determine models."""
    if name == "MODEL_FAMILY_FROM_NAME":
        return {
            model_name: model.family.__name__
            for model_name, model in MODEL_REGISTRY.items()
        }
    elif name == "MODEL_NAMES":
        return list(MODEL_REGISTRY.keys())
    else:
        raise AttributeError(
            f"Not an attribute of `vak.models.registry`: {name}"
        )
