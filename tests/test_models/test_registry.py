import inspect

import pytest

import vak.models.registry
from .conftest import (
    MockModel,
    MockEncoderDecoderModel,
    MockModelFamily,
    UnregisteredMockModelFamily,
)
from .test_definition import (
    TweetyNetDefinition,
)


def test_model_family():
    """Test that :func:`vak.models.registry.model_family`
    adds a model family to the registry"""
    # we make this copy so that we don't register UnregisteredMockModelFamily;
    # we need that class to stay unregistered for other tests
    ModelFamilyCopy = type('ModelFamilyCopy',
                           UnregisteredMockModelFamily.__bases__,
                           dict(UnregisteredMockModelFamily.__dict__))
    assert ModelFamilyCopy.__name__ not in vak.models.registry.MODEL_FAMILY_REGISTRY
    vak.models.registry.model_family(ModelFamilyCopy)
    assert ModelFamilyCopy.__name__ in vak.models.registry.MODEL_FAMILY_REGISTRY
    assert vak.models.registry.MODEL_FAMILY_REGISTRY[ModelFamilyCopy.__name__] == ModelFamilyCopy


@pytest.mark.parametrize(
    'family, definition',
    [
        (MockModelFamily, MockModel),
        (MockModelFamily, MockEncoderDecoderModel),
    ]
)
def test_register_model(family, definition):
    """Test that :func:`vak.models.registry.register_model`
    adds a model to the registry"""
    # to set up, we repeat what :func:`vak.models.decorator.model` does
    model_name = definition.__name__
    model_factory = vak.models.factory.ModelFactory(
        definition,
        family
    )
    model_factory.__name__ = model_name

    assert model_name not in vak.models.registry.MODEL_REGISTRY
    vak.models.registry.register_model(model_factory)
    assert model_name in vak.models.registry.MODEL_REGISTRY
    assert vak.models.registry.MODEL_REGISTRY[model_name] == model_factory
    del vak.models.registry.MODEL_REGISTRY[model_name]  # so this test doesn't fail for the second case


def test_register_model_raises_family():
    """Test that :func:`vak.models.registry.register_model`
    raises an error if parent class is not in model_family_classes"""
    # to set up, we repeat what :func:`vak.models.decorator.model` does,
    # but notice that we use an unregistered model family
    model_factory = vak.models.ModelFactory(
        MockModel,
        UnregisteredMockModelFamily,
    )

    with pytest.raises(TypeError):
        vak.models.registry.register_model(model_factory)


@pytest.mark.parametrize(
    'family, definition',
    [
        (vak.models.FrameClassificationModel, TweetyNetDefinition),
    ]
)
def test_register_model_raises_registered(family, definition):
    """Test that :func:`vak.models.registry.register_model`
    raises an error if a class is already registered"""
    # to set up, we repeat what :func:`vak.models.decorator.model` does
    model_factory = vak.models.ModelFactory(
        definition,
        family
    )

    # NOTE we replace 'Definition' with an empty string
    # so that the name clashes with an existing model name
    model_factory.__name__ = definition.__name__.replace('Definition', '')

    with pytest.raises(ValueError):
        vak.models.registry.register_model(model_factory)


def test___get_attr__MODEL_FAMILY_FROM_NAME():
    assert hasattr(vak.models.registry, 'MODEL_FAMILY_FROM_NAME')

    model_family_from_name_dict = getattr(vak.models.registry, 'MODEL_FAMILY_FROM_NAME')
    assert isinstance(model_family_from_name_dict, dict)

    for model_name, model_factory in vak.models.registry.MODEL_REGISTRY.items():
        model_family = model_factory.family
        family_name = model_family.__name__
        assert model_family_from_name_dict[model_name] == family_name


def test___get_attr__MODEL_NAMES():
    assert hasattr(vak.models.registry, 'MODEL_NAMES')
    model_names_list = getattr(vak.models.registry, 'MODEL_NAMES')
    assert isinstance(model_names_list, list)
    for model_name in vak.models.registry.MODEL_REGISTRY.keys():
            assert model_name in model_names_list
