import copy

import pytest

import torch
from vak import metrics
from vak.nets import TweetyNet

import vak.models

from .conftest import (
    MockEncoder,
    MockDecoder,
)


class TweetyNetDefinition:
    """Redefine here to test that ``vak.models.definition.validate``
    actually works on classes we care about.

    Can't use the class itself in the ``vak.models.tweetynet``
    because that's already decorated with ``vak.models.decorator.model``.
    """
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        'optimizer':
            {'lr': 0.003}
    }


class MissingClassVarModelDefinition:
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    # no metrics, missing class var


class ExtraClassVarModelDefinition:
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    extra_class_variable = TweetyNet


class InvalidNetworkTypeModelDefinition:
    network = torch.optim.Adam  # should be torch.nn.Module
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}


class InvalidNetworkDictKeyModelDefinition:
    network = {torch.optim.Adam: torch.optim.Adam}  # key should be a string
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}


class InvalidNetworkDictValueModelDefinition:
    network = {'adamnet': None}  # value should be torch.nn.Module
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}


class InvalidLossTypeModelDefinition:
    network = TweetyNet
    loss = None  # should be torch.nn.Module or callable
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}


class InvalidOptimTypeModelDefinition:
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = TweetyNet  # should be a torch.optim subclass
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}


class InvalidMetricsTypeModelDefinition:
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = metrics.Accuracy  # metrics should be a dict


class InvalidMetricsDictKeyModelDefinition:
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {metrics.Accuracy: metrics.Accuracy}  # key should  be str


class InvalidMetricsDictValueModelDefinition:
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': None}  # should be callable


class InvalidDefaultConfigNetworkKwargDefinition:
    """Definition with invalid keyword arg in 'network' config"""
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        # invalid keyword arg should raise error
        'network': {'invalid_keyword_arg': 100000}
    }


class InvalidDefaultConfigNetworkDictKeyDefinition:
    """Definition that specifies 'network' as a dict
    but the default_config' has a key that is not any of those network names"""
    network = {'MockEncoder': MockEncoder, 'MockDecoder': MockDecoder}
    loss = torch.nn.TripletMarginWithDistanceLoss
    optimizer = torch.optim.Adam
    metrics = {
        'loss': torch.nn.TripletMarginWithDistanceLoss
    }
    default_config = {
        # invalid key that should be network name
        'invalid_network_name': {}
    }


class InvalidDefaultConfigNetworkDictKwargDefinition:
    """Definition that specifies 'network' as a dict
    but the default_config' has a kwarg that is not valid
    for one of the networks"""
    network = {'MockEncoder': MockEncoder, 'MockDecoder': MockDecoder}
    loss = torch.nn.TripletMarginWithDistanceLoss
    optimizer = torch.optim.Adam
    metrics = {
        'loss': torch.nn.TripletMarginWithDistanceLoss
    }
    default_config = {
        # invalid key that should be network name
        'MockEncoder': {'invalid_kwarg'},
        'MockDecoder': {},
    }


class InvalidDefaultConfigLossIsFunctionButKwargsModelDefinition:
    """Definition with default config that specifies kwargs for loss,
    but loss is a function (so it can't take kwargs on __init__,
    there's no __init__)."""
    network = TweetyNet
    loss = lambda y, y_pred: y == y_pred
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        'loss': {'invalid_kwarg': 10000}
    }


class InvalidDefaultConfigLossKwarg:
    """Definition with invalid keyword arg in 'loss' config"""
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        # invalid keyword arg should raise error
        'loss': {'invalid_keyword_arg': 100000}
    }


class InvalidDefaultConfigOptimizerKwarg:
    """Definition with invalid keyword arg in 'optimizer' config"""
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        # invalid keyword arg should raise error
        'optimizer': {'invalid_keyword_arg': 100000}
    }


class InvalidDefaultConfigMetricName:
    """Definition with invalid metric name in 'metric' config"""
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        # invalid metric name should raise error
        'metrics': {'invalid_metric_name': {}}
    }


class InvalidDefaultConfigMetricKwarg:
    """Definition with invalid metric name in 'metric' config"""
    network = TweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        # invalid keyword arg should raise error
        'metrics': {'acc': {'invalid_kwarg': 10000}}
    }


class TestModelDefinition:
    """Test the class ``vak.models.definition.ModelDefinition``"""

    @pytest.mark.parametrize(
        'definition',
        [
            TweetyNetDefinition,
        ]
    )
    def test_validate(self, definition):
        """Test that ``ModelDefinition.validate`` works as expected.

        This is the main method we need to test on this class,
        since it's used by other functions,
        e.g. ``vak.models.make.make`` to validate a class
        passed in that claims to be a model definition.
        """
        before = copy.deepcopy(definition)
        after = vak.models.definition.validate(definition)
        assert after is definition

        if not hasattr(before, 'default_config'):
            assert hasattr(after, 'default_config')
            assert isinstance(after.default_config, dict)
            assert after.default_config == vak.models.definition.DEFAULT_DEFAULT_CONFIG

        elif hasattr(before, 'default_config'):
            for key in vak.models.definition.DEFAULT_DEFAULT_CONFIG:
                if key not in before.default_config:
                    assert after.default_config[key] == {}
                else:
                    assert after.default_config[key] == before.default_config[key]

    @pytest.mark.parametrize(
        'definition, expected_error',
        [
            (MissingClassVarModelDefinition, ValueError),
            (ExtraClassVarModelDefinition, ValueError),
            (InvalidNetworkTypeModelDefinition, TypeError),
            (InvalidNetworkDictKeyModelDefinition, TypeError),
            (InvalidNetworkDictValueModelDefinition, TypeError),
            (InvalidLossTypeModelDefinition, TypeError),
            (InvalidOptimTypeModelDefinition, TypeError),
            (InvalidMetricsTypeModelDefinition, TypeError),
            (InvalidMetricsDictKeyModelDefinition, TypeError),
            (InvalidMetricsDictValueModelDefinition, TypeError),
            (InvalidDefaultConfigNetworkKwargDefinition, ValueError),
            (InvalidDefaultConfigNetworkDictKeyDefinition, ValueError),
            (InvalidDefaultConfigNetworkDictKwargDefinition, ValueError),
            (InvalidDefaultConfigLossIsFunctionButKwargsModelDefinition, ValueError),
            (InvalidDefaultConfigLossKwarg, ValueError),
            (InvalidDefaultConfigOptimizerKwarg, ValueError),
            (InvalidDefaultConfigMetricName, ValueError),
            (InvalidDefaultConfigMetricKwarg, ValueError),
        ]
    )
    def test_validate_raises(self, definition, expected_error):
        with pytest.raises(expected_error):
            vak.models.definition.validate(definition)
