import pytest

import vak.models

from .conftest import MockModel, MockModelFamily, MockEncoderDecoderModel
from .test_definition import TweetyNetDefinition as TweetyNet
from .test_definition import TeenyTweetyNetDefinition as TeenyTweetyNet

from .test_definition import (
    MissingClassVarModelDefinition,
    ExtraClassVarModelDefinition,
    InvalidNetworkTypeModelDefinition,
    InvalidNetworkDictKeyModelDefinition,
    InvalidNetworkDictValueModelDefinition,
    InvalidLossTypeModelDefinition,
    InvalidOptimTypeModelDefinition,
    InvalidMetricsTypeModelDefinition,
    InvalidMetricsDictKeyModelDefinition,
    InvalidMetricsDictValueModelDefinition,
)


TweetyNet.__name__ = 'TweetyNet'
TeenyTweetyNet.__name__ = 'TeenyTweetyNet'


@pytest.mark.parametrize(
    'definition, family, expected_name',
    [
        (MockModel,
         MockModelFamily,
         'MockModel'),
        (MockEncoderDecoderModel,
         MockModelFamily,
         'MockEncoderDecoderModel'),
    ]
)
def test_model(definition, family, expected_name):
    """Test that :func:`vak.models.decorator.model` decorator
    returns a subclass of the specified model family,
    and has the expected name"""
    model_class = vak.models.decorator.model(family)(definition)
    assert issubclass(model_class, family)
    assert model_class.__name__ == expected_name


@pytest.mark.parametrize(
    'definition',
    [
        MissingClassVarModelDefinition,
        ExtraClassVarModelDefinition,
        InvalidNetworkTypeModelDefinition,
        InvalidNetworkDictKeyModelDefinition,
        InvalidNetworkDictValueModelDefinition,
        InvalidLossTypeModelDefinition,
        InvalidOptimTypeModelDefinition,
        InvalidMetricsTypeModelDefinition,
        InvalidMetricsDictKeyModelDefinition,
        InvalidMetricsDictValueModelDefinition,
    ]
)
def test_model_raises(definition):
    with pytest.raises(vak.models.decorator.ModelDefinitionValidationError):
        model_class = vak.models.decorator.model(vak.models.base.Model)(definition)
