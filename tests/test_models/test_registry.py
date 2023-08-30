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
    attributes = dict(family.__dict__)
    attributes.update({"definition": definition})
    subclass_name = definition.__name__
    subclass = type(subclass_name, (family,), attributes)
    subclass.__module__ = definition.__module__

    assert subclass_name not in vak.models.registry.MODEL_REGISTRY
    vak.models.registry.register_model(subclass)
    assert subclass_name in vak.models.registry.MODEL_REGISTRY
    assert vak.models.registry.MODEL_REGISTRY[subclass_name] == subclass
    del vak.models.registry.MODEL_REGISTRY[subclass_name]  # so this test doesn't fail for the second case


def test_register_model_raises_family():
    """Test that :func:`vak.models.registry.register_model`
    raises an error if parent class is not in model_family_classes"""
    # to set up, we repeat what :func:`vak.models.decorator.model` does,
    # but notice that we use an unregistered model family
    attributes = dict(UnregisteredMockModelFamily.__dict__)
    attributes.update({"definition": MockModel})
    subclass_name = MockModel.__name__
    subclass = type(subclass_name, (UnregisteredMockModelFamily,), attributes)
    subclass.__module__ = MockModel.__module__

    with pytest.raises(TypeError):
        vak.models.registry.register_model(subclass)


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
    attributes = dict(family.__dict__)
    attributes.update({"definition": definition})
    # NOTE we replace 'Definition' with an empty string
    # so that the name clashes with an existing model name
    subclass_name = definition.__name__.replace('Definition', '')
    subclass = type(subclass_name, (family,), attributes)
    subclass.__module__ = definition.__module__

    with pytest.raises(ValueError):
        vak.models.registry.register_model(subclass)


def test___get_attr__MODEL_FAMILY_FROM_NAME():
    assert hasattr(vak.models.registry, 'MODEL_FAMILY_FROM_NAME')
    attr = getattr(vak.models.registry, 'MODEL_FAMILY_FROM_NAME')
    assert isinstance(attr, dict)
    for model_name, model_class in vak.models.registry.MODEL_REGISTRY.items():
        model_parent_class = inspect.getmro(model_class)[1]
        family_name = model_parent_class.__name__
        assert attr[model_name] == family_name


def test___get_attr__MODEL_NAMES():
    assert hasattr(vak.models.registry, 'MODEL_NAMES')
    attr = getattr(vak.models.registry, 'MODEL_NAMES')
    assert isinstance(attr, list)
    for model_name in vak.models.registry.MODEL_REGISTRY.keys():
            assert model_name in attr
