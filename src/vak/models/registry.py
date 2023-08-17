"""Registry for models.

Makes it possible to register a model declared outside of ``vak``
with a decorator, so that the model can be used at runtime.
"""
from __future__ import annotations

import inspect
from typing import Any, Type

from .base import Model

MODEL_FAMILY_REGISTRY = {}


def model_family(family_class: Type) -> None:
    """Decorator that adds a class to the registry of model families."""
    if family_class not in Model.__subclasses__():
        raise TypeError(
            "The ``family_class`` provided to the `vak.models.model_family` decorator"
            "must be a subclass of `vak.models.base.Model`, "
            f"but the class specified is not: {family_class}. "
            f"Subclasses of `vak.models.base.Model` are: {Model.__subclasses__()}"
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


def register_model(model_class: Type) -> Type:
    """Decorator that registers a model in the model registry.

    This function is called by :func:`vak.models.decorator.model`,
    that creates a model class from a model definition.
    So you will not usually need to use this decorator directly,
    and should prefer to use :func:`vak.models.decorator.model` instead.
    """
    model_family_classes = list(MODEL_FAMILY_REGISTRY.values())
    model_parent_class = inspect.getmro(model_class)[1]
    if model_parent_class not in model_family_classes:
        raise TypeError(
            "The parent class of ``model_class`` passed to the ``model`` decorator "
            f"is not recognized as a model family. Class was: {model_class} and "
            f"parent is {model_parent_class}, as determined with "
            f"``inspect.getmro(model_class)[1]``. "
            f"Please specify a class that is a sub-class of a model family. "
            f"Valid model family classes are: {model_family_classes}"
        )

    model_name = model_class.__name__
    if model_name in MODEL_REGISTRY:
        raise ValueError(
            f"Attempted to register a model family with the name '{model_name}', "
            f"but this name is already in the registry.\n"
        )

    MODEL_REGISTRY[model_name] = model_class
    # need to return class after we register it or we replace it with None
    # when this function is used as a decorator
    return model_class


def __getattr__(name: str) -> Any:
    """Module-level __getattr__ function that we use to dynamically determine models."""
    if name == "MODEL_FAMILY_FROM_NAME":
        model_name_family_name_map = {}
        for model_name, model_class in MODEL_REGISTRY.items():
            model_parent_class = inspect.getmro(model_class)[1]
            family_name = model_parent_class.__name__
            model_name_family_name_map[model_name] = family_name
        return model_name_family_name_map
    elif name == "MODEL_NAMES":
        return list(
            MODEL_REGISTRY.keys()
        )
    else:
        raise AttributeError(
            f"Not an attribute of `vak.models.registry`: {name}"
        )
