"""Decorator that makes a model class,
given a definition of the model,
and another class that represents a
family of models that the new model belongs to.

The function returns a newly-created subclass
of the class representing the family of models.
The subclass can then be instantiated
and have all model methods.
"""
from __future__ import annotations

from typing import Type

from .base import Model
from .definition import validate as validate_definition
from .registry import register_model


class ModelDefinitionValidationError(Exception):
    """Exception raised when validating a model
    definition fails.

    Used by :func:`vak.models.decorator.model` decorator.
    """

    pass


def model(family: Type[Model]):
    """Decorator that makes a model class,
    given a definition of the model,
    and another class that represents a
    family of models that the new model belongs to.

    Returns a newly-created subclass
    of the class representing the family of models.
    The subclass can then be instantiated
    and have all model methods.

    Parameters
    ----------
    definition : type
        The definition of the new model that will be made.
        A class with all the class variables required
        by :func:`vak.models.definition.validate`.
        See docstring of that function for specification.
    family : subclass of vak.models.Model
        The class representing the family of models
        that the new model will belong to.
        E.g., :class:`vak.models.FrameClassificationModel`.

    Returns
    -------
    model : type
        A sub-class of ``model_family``,
        with attribute ``definition``,
        that will be used when making
        new instances of the model.
    """

    def _model(definition: Type):
        if not issubclass(family, Model):
            raise TypeError(
                "The ``family`` argument to the ``vak.models.model`` decorator"
                "should be a subclass of ``vak.models.base.Model``,"
                f"but the type was: {type(family)}, "
                "which was not recognized as a subclass "
                "of ``vak.models.base.Model``."
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

        attributes = dict(family.__dict__)
        attributes.update({"definition": definition})
        subclass_name = definition.__name__
        subclass = type(subclass_name, (family,), attributes)
        subclass.__module__ = definition.__module__

        # finally, add model to registry
        register_model(subclass)

        return subclass

    return _model
