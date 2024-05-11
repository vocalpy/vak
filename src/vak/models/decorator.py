"""Decorator that makes a :class:`vak.models.ModelFactory`,
given a definition of the model,
and a :class:`lightning.LightningModule` that represents a
family of models that the new model belongs to.

The function returns a new instance of :class:`vak.models.ModelFactory`,
that can create new instances of the model with its
:meth:`~:class:`vak.models.ModelFactory.from_config` and
:meth:`~:class:`vak.models.ModelFactory.from_instances` methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

import lightning

from .registry import register_model

if TYPE_CHECKING:
    from .factory import ModelFactory


class ModelDefinitionValidationError(Exception):
    """Exception raised when validating a model
    definition fails.

    Used by :func:`vak.models.decorator.model` decorator.
    """

    pass


def model(family: lightning.pytorch.LightningModule):
    """Decorator that makes a :class:`vak.models.ModelFactory`,
    given a definition of the model,
    and a :class:`lightning.LightningModule` that represents a
    family of models that the new model belongs to.

    The function returns a new instance of :class:`vak.models.ModelFactory`,
    that can create new instances of the model with its
    :meth:`~:class:`vak.models.ModelFactory.from_config` and
    :meth:`~:class:`vak.models.ModelFactory.from_instances` methods.

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
    family : lightning.LightningModule
        The class representing the family of models
        that the new model will belong to.
        E.g., :class:`vak.models.FrameClassificationModel`.
        Should be a subclass of :class:`lightning.LightningModule`
        that was registered with the
        :func:`vak.models.registry.model_family` decorator.

    Returns
    -------
    model_factory : vak.models.ModelFactory
        An instance of :class:`~vak.models.ModelFactory`,
        with attribute ``definition`` and ``family``,
        that will be used when making
        new instances of the model by calling the
        :meth:`~vak.models.ModelFactory.from_config` method
        or the :meth:`~:class:`vak.models.ModelFactory.from_instances` method.
    """

    def _model(definition: Type) -> ModelFactory:
        from .factory import ModelFactory  # avoid circular import

        model_factory = ModelFactory(definition, family)
        model_factory.__name__ = definition.__name__
        model_factory.__doc__ = definition.__doc__
        model_factory.__module__ = definition.__module__
        register_model(model_factory)
        return model_factory

    return _model
