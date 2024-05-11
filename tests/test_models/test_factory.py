import inspect
import itertools

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
from .test_tweetynet import LABELMAPS, INPUT_SHAPES

MODEL_DEFINITION_CLASS_VARS = (
    'network',
    'loss',
    'optimizer',
    'metrics',
    'default_config'
)

mock_net_instance = MockNetwork()

TEST_VALIDATE_RAISES_ARGVALS = [
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

# pytest.mark.parametrize vals for test_init_with_definition
MODEL_DEFS = (
    TweetyNetDefinition,
)

TEST_WITH_FRAME_CLASSIFICATION_ARGVALS = itertools.product(LABELMAPS, INPUT_SHAPES, MODEL_DEFS)

MOCK_INPUT_SHAPE = torch.Size([1, 128, 44])


class ConvEncoderUMAPDefinition:
    network = {"encoder": vak.nets.ConvEncoder}
    loss = vak.nn.UmapLoss
    optimizer = torch.optim.AdamW
    metrics = {
        "acc": vak.metrics.Accuracy,
        "levenshtein": vak.metrics.Levenshtein,
        "character_error_rate": vak.metrics.CharacterErrorRate,
        "loss": torch.nn.CrossEntropyLoss,
    }
    default_config = {
        "optimizer": {"lr": 1e-3},
    }


class TestModelFactory:
    def test_init_no_definition_raises(self):
        """Test that initializing a Model instance without a definition or family raises a ValueError."""
        with pytest.raises(TypeError):
            vak.models.factory.ModelFactory()

    def test_init_invalid_definition_raises(self):
        """Test that initializing a Model instance with an invalid definition raises a ValueError."""
        with pytest.raises(vak.models.decorator.ModelDefinitionValidationError):
            vak.models.factory.ModelFactory(
                definition=InvalidMetricsDictKeyModelDefinition,
                family=MockModelFamily,
            )

    @pytest.mark.parametrize(
        'definition, kwargs',
        [
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
    )
    def test_validate_instances_or_get_default(self, definition, kwargs):
        model = vak.models.factory.ModelFactory(
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
        TEST_VALIDATE_RAISES_ARGVALS
    )
    def test_validate_instances_or_get_default_raises(self, definition, kwargs, expected_exception):
        """Test that :meth:`validate_instances_or_get_default` raises errors as expected given input arguments.

        Note that this should happen from ``validate_instances_or_get_default`` calling ``validate_init``,
        so here we test that this is happening inside ``validate_instances_or_get_default``.
        Next method tests ``validate_init`` directly.
        """
        model = vak.models.factory.ModelFactory(
            definition,
            MockModelFamily,
        )
        with pytest.raises(expected_exception):
            model.validate_instances_or_get_default(**kwargs)

    @pytest.mark.parametrize(
        'definition, kwargs, expected_exception',
        TEST_VALIDATE_RAISES_ARGVALS
    )
    def test_validate_init_raises(self, definition, kwargs, expected_exception):
        """Test that ``validate_init`` raises errors as expected"""
        model = vak.models.factory.ModelFactory(
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
        model = vak.models.factory.ModelFactory(
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

    @pytest.mark.parametrize(
            'labelmap, input_shape, definition',
            TEST_WITH_FRAME_CLASSIFICATION_ARGVALS
    )
    def test_from_config_frame_classification(self, labelmap, input_shape, definition):
        model_factory = vak.models.factory.ModelFactory(
            definition,
            vak.models.FrameClassificationModel,
        )
        num_input_channels, num_freqbins = input_shape[0], input_shape[1]
        # network has required args that need to be determined dynamically
        network = definition.network(len(labelmap), num_input_channels, num_freqbins)
        model = model_factory.from_instances(network=network, labelmap=labelmap)

        # now test that attributes are what we expect
        assert isinstance(model, vak.models.FrameClassificationModel)
        for attr in ('network', 'loss', 'optimizer', 'metrics'):
            assert hasattr(model, attr)
            model_attr = getattr(model, attr)
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
        'definition',
        [
            TweetyNetDefinition,
        ]
    )
    def test_from_config_with_frame_classification(self, definition, specific_config_toml_path):
        model_name = definition.__name__.replace('Definition', '')
        toml_path = specific_config_toml_path('train', model_name, audio_format='cbin', annot_format='notmat')
        cfg = vak.config.Config.from_toml_path(toml_path)

        # stuff we need just to be able to instantiate network
        labelmap = vak.common.labels.to_map(cfg.prep.labelset, map_background=True)

        model_factory = vak.models.factory.ModelFactory(
            definition,
            vak.models.FrameClassificationModel,
        )

        config = cfg.train.model.asdict()
        num_input_channels, num_freqbins = MOCK_INPUT_SHAPE[0], MOCK_INPUT_SHAPE[1]

        config["network"].update(
            num_classes=len(labelmap),
            num_input_channels=num_input_channels,
            num_freqbins=num_freqbins
        )

        model = model_factory.from_config(config=config, labelmap=labelmap)
        assert isinstance(model, vak.models.FrameClassificationModel)

        # below, we can only test the config kwargs that actually end up as attributes
        # so we use `if hasattr` before checking
        if 'network' in config:
            if inspect.isclass(definition.network):
                for network_kwarg, network_kwargval in config['network'].items():
                    if hasattr(model.network, network_kwarg):
                        assert getattr(model.network, network_kwarg) == network_kwargval
            elif isinstance(definition.network, dict):
                for net_name, net_kwargs in config['network'].items():
                    for network_kwarg, network_kwargval in net_kwargs.items():
                        if hasattr(model.network[net_name], network_kwarg):
                            assert getattr(model.network[net_name], network_kwarg) == network_kwargval

        if 'loss' in config:
            for loss_kwarg, loss_kwargval in config['loss'].items():
                if hasattr(model.loss, loss_kwarg):
                    assert getattr(model.loss, loss_kwarg) == loss_kwargval

        if 'optimizer' in config:
            for optimizer_kwarg, optimizer_kwargval in config['optimizer'].items():
                if optimizer_kwarg in model.optimizer.param_groups[0]:
                    assert model.optimizer.param_groups[0][optimizer_kwarg] == optimizer_kwargval

        if 'metrics' in config:
            for metric_name, metric_kwargs in config['metrics'].items():
                assert metric_name in model.metrics
                for metric_kwarg, metric_kwargval in metric_kwargs.items():
                    if hasattr(model.metrics[metric_name], metric_kwarg):
                        assert getattr(model.metrics[metric_name], metric_kwarg) == metric_kwargval

    @pytest.mark.parametrize(
        'input_shape, definition',
        [
            ((1, 128, 128), ConvEncoderUMAPDefinition),
        ]
    )
    def test_from_instances_parametric_umap(
            self,
            input_shape,
            definition,
    ):
        network = {'encoder': vak.nets.ConvEncoder(input_shape)}

        model_factory = vak.models.ModelFactory(
            definition,
            vak.models.ParametricUMAPModel,
        )
        model = model_factory.from_instances(network=network)

        # now test that attributes are what we expect
        assert isinstance(model, vak.models.ParametricUMAPModel)
        for attr in ('network', 'loss', 'optimizer', 'metrics'):
            assert hasattr(model, attr)
            model_attr = getattr(model, attr)
            definition_attr = getattr(definition, attr)
            if inspect.isclass(definition_attr):
                assert isinstance(model_attr, definition_attr)
            elif isinstance(definition_attr, dict):
                assert isinstance(model_attr, (dict, torch.nn.ModuleDict))
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
        'input_shape, definition',
        [
            ((1, 128, 128), ConvEncoderUMAPDefinition),
        ]
    )
    def test_from_config_with_parametric_umap(
            self,
            input_shape,
            definition,
            specific_config_toml_path,
    ):
        model_name = definition.__name__.replace('Definition', '')
        toml_path = specific_config_toml_path('train', model_name, audio_format='cbin', annot_format='notmat')
        cfg = vak.config.Config.from_toml_path(toml_path)

        model_factory = vak.models.ModelFactory(
            definition,
            vak.models.ParametricUMAPModel,
        )

        config = cfg.train.model.asdict()
        config["network"]["encoder"]["input_shape"] = input_shape

        model = model_factory.from_config(config=config)
        assert isinstance(model, vak.models.ParametricUMAPModel)

        if 'network' in config:
            if inspect.isclass(definition.network):
                for network_kwarg, network_kwargval in config['network'].items():
                    assert hasattr(model.network, network_kwarg)
                    assert getattr(model.network, network_kwarg) == network_kwargval
            elif isinstance(definition.network, dict):
                for net_name, net_kwargs in config['network'].items():
                    for network_kwarg, network_kwargval in net_kwargs.items():
                        network = model.network[net_name]
                        if hasattr(network, network_kwarg):
                            assert getattr(network, network_kwarg) == network_kwargval

        if 'loss' in config:
            for loss_kwarg, loss_kwargval in config['loss'].items():
                assert hasattr(model.loss, loss_kwarg)
                assert getattr(model.loss, loss_kwarg) == loss_kwargval

        if 'optimizer' in config:
            for optimizer_kwarg, optimizer_kwargval in config['optimizer'].items():
                assert optimizer_kwarg in model.optimizer.param_groups[0]
                assert model.optimizer.param_groups[0][optimizer_kwarg] == optimizer_kwargval

        if 'metrics' in config:
            for metric_name, metric_kwargs in config['metrics'].items():
                assert metric_name in model.metrics
                for metric_kwarg, metric_kwargval in metric_kwargs.items():
                    assert hasattr(model.metrics[metric_name], metric_kwarg)
                    assert getattr(model.metrics[metric_name], metric_kwarg) == metric_kwargval
