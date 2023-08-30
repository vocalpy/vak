import copy
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

    @pytest.mark.parametrize(
        'definition, kwargs',
        TEST_INIT_ARGVALS,
    )
    def test_init(self,
                  definition,
                  kwargs,
                  monkeypatch):
        """Test Model.__init__ works as expected"""
        # monkeypatch a definition so we can test __init__
        definition = vak.models.definition.validate(definition)
        monkeypatch.setattr(
            vak.models.base.Model, 'definition', definition, raising=False
        )

        # actually instantiate model
        if kwargs:
            model = vak.models.base.Model(**kwargs)
        else:
            model = vak.models.base.Model()

        # now test that attributes are what we expect
        assert isinstance(model, vak.models.base.Model)
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

    def test_init_no_definition_raises(self):
        """Test that initializing a Model instance without a definition raises a ValueError."""
        with pytest.raises(ValueError):
            vak.models.base.Model()

    def test_init_invalid_definition_raises(self, monkeypatch):
        """Test that initializing a Model instance with an invalid definition raises a ValueError."""
        monkeypatch.setattr(
            vak.models.base.Model, 'definition', InvalidMetricsDictKeyModelDefinition, raising=False
        )
        with pytest.raises(TypeError):
            vak.models.base.Model()

    @pytest.mark.parametrize(
        'definition, kwargs, expected_exception',
        TEST_INIT_RAISES_ARGVALS
    )
    def test_init_raises(self, definition, kwargs, expected_exception, monkeypatch):
        """Test that init raises errors as expected given input arguments.

        Note that this should happen from ``__init__`` calling ``validate_init``,
        so here we test that this is happening inside ``__init__``.
        Next method tests ``validate_init`` directly.
        """
        # monkeypatch a definition so we can test __init__
        monkeypatch.setattr(
            # we just always use TweetyNetDefinition here since we just want to test that a mismatch raises
            vak.models.base.Model, 'definition', definition, raising=False
        )
        with pytest.raises(expected_exception):
            vak.models.base.Model(**kwargs)

    @pytest.mark.parametrize(
        'definition, kwargs, expected_exception',
        TEST_INIT_RAISES_ARGVALS
    )
    def test_validate_init_raises(self, definition, kwargs, expected_exception, monkeypatch):
        """Test that ``validate_init`` raises errors as expected"""
        # monkeypatch a definition so we can test __init__
        monkeypatch.setattr(
            # we just always use TweetyNetDefinition here since we just want to test that a mismatch raises
            vak.models.base.Model, 'definition', definition, raising=False
        )
        with pytest.raises(expected_exception):
            vak.models.base.Model.validate_init(**kwargs)

    MODEL_DEFINITION_MAP = {
        'TweetyNet': TweetyNetDefinition,
    }

    @pytest.mark.parametrize(
        'model_name',
        [
            'TweetyNet',
        ]
    )
    def test_load_state_dict_from_path(self,
                                       model_name,
                                       # our fixtures
                                       specific_config,
                                       # pytest fixtures
                                       monkeypatch,
                                       device
                                       ):
        """Smoke test that makes sure ``load_state_dict_from_path`` runs without failure.

        We use actual model definitions here so we can test with real checkpoints.
        """
        definition = self.MODEL_DEFINITION_MAP[model_name]
        train_toml_path = specific_config('train', model_name, audio_format='cbin', annot_format='notmat')
        train_cfg = vak.config.parse.from_toml_path(train_toml_path)

        # stuff we need just to be able to instantiate network
        labelmap = vak.common.labels.to_map(train_cfg.prep.labelset, map_unlabeled=True)
        transform, target_transform = vak.transforms.defaults.get_default_transform(
            model_name,
            "train",
            transform_kwargs={},
        )
        train_dataset = vak.datasets.frame_classification.WindowDataset.from_dataset_path(
            dataset_path=train_cfg.train.dataset_path,
            split="train",
            window_size=train_cfg.train.train_dataset_params['window_size'],
            transform=transform,
            target_transform=target_transform,
        )
        input_shape = train_dataset.shape
        num_input_channels = input_shape[-3]
        num_freqbins = input_shape[-2]

        monkeypatch.setattr(
            vak.models.base.Model, 'definition', definition, raising=False
        )
        # network is the one thing that has required args
        # and we also need to use its config from the toml file
        model_config = vak.config.model.config_from_toml_path(train_toml_path, model_name)
        network = definition.network(num_classes=len(labelmap),
                                     num_input_channels=num_input_channels,
                                     num_freqbins=num_freqbins,
                                     **model_config['network'])
        model = vak.models.base.Model(network=network)
        model.to(device)

        eval_toml_path = specific_config('eval', model_name, audio_format='cbin', annot_format='notmat')
        eval_cfg = vak.config.parse.from_toml_path(eval_toml_path)
        checkpoint_path = eval_cfg.eval.checkpoint_path

        # ---- actually test method
        sd_before = copy.deepcopy(model.state_dict())
        sd_before = {
            k: v.to(device) for k, v in sd_before.items()
        }
        ckpt = torch.load(checkpoint_path)
        sd_to_be_loaded = ckpt['state_dict']
        sd_to_be_loaded = {
            k: v.to(device) for k, v in sd_to_be_loaded.items()
        }

        model.load_state_dict_from_path(checkpoint_path)

        assert not all([
            torch.all(torch.eq(val, before_val))
            for val, before_val in zip(model.state_dict().values(), sd_before.values())]
        )
        assert all([
            torch.all(torch.eq(val, before_val))
            for val, before_val in zip(model.state_dict().values(), sd_to_be_loaded.values())]
        )

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
                         monkeypatch,
                         ):
        monkeypatch.setattr(
            vak.models.base.Model, 'definition', definition, raising=False
        )

        model = vak.models.base.Model.from_config(config)

        assert isinstance(model, vak.models.base.Model)

        if 'network' in config:
            if inspect.isclass(definition.network):
                for network_kwarg, network_kwargval in config['network'].items():
                    assert hasattr(model.network, network_kwarg)
                    assert getattr(model.network, network_kwarg) == network_kwargval
            elif isinstance(definition.network, dict):
                for net_name, net_kwargs in config['network'].items():
                    for network_kwarg, network_kwargval in net_kwargs.items():
                        assert hasattr(model.network[net_name], network_kwarg)
                        assert getattr(model.network[net_name], network_kwarg) == network_kwargval

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
