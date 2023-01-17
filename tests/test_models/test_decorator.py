import pytest

import vak.models

from .test_base import MockModel, MockEncoderDecoderModel
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
         vak.models.Model,
         'MockModel'),
        (MockEncoderDecoderModel,
         vak.models.Model,
         'MockEncoderDecoderModel'),
        (TweetyNet,
         vak.models.WindowedFrameClassificationModel,
         'TweetyNet'),
        (TeenyTweetyNet,
         vak.models.WindowedFrameClassificationModel,
         'TeenyTweetyNet'),
    ]
)
def test_model(definition, family, expected_name):
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
