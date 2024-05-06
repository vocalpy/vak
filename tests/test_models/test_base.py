import inspect

import pytest
import torch

import vak

from .conftest import (
    MockAcc,
    MockDecoder,
    MockEncoder,
    MockEncoderDecoderModel,
    MockModel,
    MockModelFamily,
    MockNetwork,
    other_loss_func,
    other_metrics_dict,
    OtherNetwork,
    OtherOptimizer,
)

from .test_definition import (
    InvalidMetricsDictKeyModelDefinition,
    TweetyNetDefinition,
)


MODEL_DEFINITION_CLASS_VARS = (
    'network',
    'loss',
    'optimizer',
    'metrics',
    'default_config'
)

mock_net_instance = MockNetwork()


TEST_INIT_ARGVALS = [
    (MockModel, None),
    (MockModel, {'network': mock_net_instance}),
    (MockModel, {'loss': torch.nn.CrossEntropyLoss()},),
    (MockModel,
     {
         'network': mock_net_instance,
         'optimizer': torch.optim.SGD(lr=0.003, params=mock_net_instance.parameters())
      }
     ),
    (MockModel,
     {'metrics':
         {
             'acc': MockAcc(),
         }
    }),
    (MockEncoderDecoderModel, None),
]

TEST_INIT_RAISES_ARGVALS = [
    (MockModel, dict(network=OtherNetwork()), TypeError),
    (MockModel, dict(loss=other_loss_func), TypeError),
    (MockModel, dict(optimizer=OtherOptimizer), TypeError),
    (MockModel, dict(metrics=other_metrics_dict), ValueError),
    (MockEncoderDecoderModel,
    # first value is wrong
     dict(network={'MockEncoder': OtherNetwork(), 'MockDecoder': MockDecoder()}),
     TypeError),
    (MockEncoderDecoderModel,
     # missng key, MockEncoder
     dict(network={'MockDecoder': MockDecoder()}),
     ValueError),
    (MockEncoderDecoderModel,
     # extra key, MockRecoder
     dict(network={'MockEncoder': MockEncoder(), 'MockDecoder': MockDecoder(), 'MockRecoder': MockNetwork()}),
     ValueError),
]


class TestModel:


    def test_init_no_definition_raises(self):
        """Test that initializing a Model instance without a definition or family raises a ValueError."""
        with pytest.raises(TypeError):
            vak.models.base.Model()

    def test_init_invalid_definition_raises(self):
        """Test that initializing a Model instance with an invalid definition raises a ValueError."""
        with pytest.raises(vak.models.decorator.ModelDefinitionValidationError):
            vak.models.base.Model(
                definition=InvalidMetricsDictKeyModelDefinition,
                family=MockModelFamily,
            )

    @pytest.mark.parametrize(
        'definition, kwargs',
        TEST_INIT_ARGVALS,
    )
    def test_validate_instances_or_get_default(self, definition, kwargs):
        model = vak.models.base.Model(
            definition,
            MockModelFamily,
        )
        # actually instantiate model
        if kwargs:
            (network,
            loss,
            optimizer,
            metrics
            ) = model.validate_instances_or_get_default(**kwargs)
        else:
            (network,
            loss,
            optimizer,
            metrics
            ) = model.validate_instances_or_get_default()

        model_attrs = {
            'network': network,
            'loss': loss,
            'optimizer': optimizer,
            'metrics': metrics,
        }
        for attr in ('network', 'loss', 'optimizer', 'metrics'):
            model_attr = model_attrs[attr]
            definition_attr = getattr(definition, attr)
            if inspect.isclass(definition_attr):
                assert isinstance(model_attr, definition_attr)
            elif isinstance(definition_attr, dict):
                assert isinstance(model_attr, dict)
                for definition_key, definition_val in definition_attr.items():
                    assert definition_key in model_attr
                    model_val = model_attr[definition_key]
                    if inspect.isclass(definition_val):
                        assert isinstance(model_val, definition_val)
                    else:
                        assert callable(definition_val)
                        assert model_val is definition_val
            else:
                # must be a function
                assert callable(model_attr)
                assert model_attr is definition_attr

    @pytest.mark.parametrize(
        'definition, kwargs, expected_exception',
        TEST_INIT_RAISES_ARGVALS
    )
    def test_validate_instances_or_get_default_raises(self, definition, kwargs, expected_exception):
        """Test that :meth:`validate_instances_or_get_default` raises errors as expected given input arguments.

        Note that this should happen from ``validate_instances_or_get_default`` calling ``validate_init``,
        so here we test that this is happening inside ``validate_instances_or_get_default``.
        Next method tests ``validate_init`` directly.
        """
        model = vak.models.base.Model(
            definition,
            MockModelFamily,
        )
        with pytest.raises(expected_exception):
            model.validate_instances_or_get_default(**kwargs)

    @pytest.mark.parametrize(
        'definition, kwargs, expected_exception',
        TEST_INIT_RAISES_ARGVALS
    )
    def test_validate_init_raises(self, definition, kwargs, expected_exception):
        """Test that ``validate_init`` raises errors as expected"""
        # monkeypatch a definition so we can test __init__
        model = vak.models.base.Model(
            definition=definition,
            family=MockModelFamily
        )
        with pytest.raises(expected_exception):
            model.validate_init(**kwargs)

    MODEL_DEFINITION_MAP = {
        'TweetyNet': TweetyNetDefinition,
    }

    @pytest.mark.parametrize(
        'definition, config',
        [
            (MockModel, {'network': {'n_classes': 10}}),
            (MockModel, {'loss': {'reduction': 'sum'}}),
            (MockModel, {
                 'network': {'n_classes': 10},
                 'optimizer': {'lr': 0.003}}
             ),
            (MockModel, {'metrics': {'acc': {'average': 'micro'}}}),
            (MockEncoderDecoderModel, {
                 'network': {
                     'MockEncoder': {'input_size': 5},
                     'MockDecoder': {'output_size': 5},
                 },
                 'optimizer': {'lr': 0.003}}
             )
        ]
    )
    def test_from_config(self,
                         definition,
                         config,
                         ):
        model = vak.models.base.Model(
            definition=definition,
            family=MockModelFamily
        )
        new_model_instance = model.from_config(config)

        assert isinstance(new_model_instance, MockModelFamily)

        if 'network' in config:
            if inspect.isclass(definition.network):
                for network_kwarg, network_kwargval in config['network'].items():
                    assert hasattr(new_model_instance.network, network_kwarg)
                    assert getattr(new_model_instance.network, network_kwarg) == network_kwargval
            elif isinstance(definition.network, dict):
                for net_name, net_kwargs in config['network'].items():
                    for network_kwarg, network_kwargval in net_kwargs.items():
                        assert hasattr(new_model_instance.network[net_name], network_kwarg)
                        assert getattr(new_model_instance.network[net_name], network_kwarg) == network_kwargval

        if 'loss' in config:
            for loss_kwarg, loss_kwargval in config['loss'].items():
                assert hasattr(new_model_instance.loss, loss_kwarg)
                assert getattr(new_model_instance.loss, loss_kwarg) == loss_kwargval

        if 'optimizer' in config:
            for optimizer_kwarg, optimizer_kwargval in config['optimizer'].items():
                assert optimizer_kwarg in new_model_instance.optimizer.param_groups[0]
                assert new_model_instance.optimizer.param_groups[0][optimizer_kwarg] == optimizer_kwargval

        if 'metrics' in config:
            for metric_name, metric_kwargs in config['metrics'].items():
                assert metric_name in new_model_instance.metrics
                for metric_kwarg, metric_kwargval in metric_kwargs.items():
                    assert hasattr(new_model_instance.metrics[metric_name], metric_kwarg)
                    assert getattr(new_model_instance.metrics[metric_name], metric_kwarg) == metric_kwargval
