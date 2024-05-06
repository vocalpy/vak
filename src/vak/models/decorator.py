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

import lightning

from .definition import validate as validate_definition
from .registry import register_model


class ModelDefinitionValidationError(Exception):
    """Exception raised when validating a model
    definition fails.

    Used by :func:`vak.models.decorator.model` decorator.
    """

    pass


def model(family: lightning.pytorch.LightningModule):
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
        See also :class:`vak.models.definition.ModelDefinition`,
        but note that it is not necessary to subclass
        :class:`~vak.models.definition.ModelDefinition` to
        define a model.
    family : lightning.pytorch.LightningModule
        The class representing the family of models
        that the new model will belong to.
        E.g., :class:`vak.models.FrameClassificationModel`.

    Returns
    -------
    model : type
        A sub-class of :class:`~vak.models.base.Model`,
        with attribute ``definition``
        and ``family``,
        that will be used when making
        new instances of the model
        by calling the
        :meth:`~vak.models.base.Model.from_config` method.
    """

    def _model(definition: Type):
        from .factory import ModelFactory  # avoid circular import

        model = ModelFactory(
            definition,
            family
        )
        model.__name__ = definition.__name__
        model.__doc__ = definition.__doc__
        model.__module__ = definition.__module__
        register_model(model)
        return model

    return _model
