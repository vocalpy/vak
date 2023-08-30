import inspect
import itertools

import pytest
import torch

import vak.models

from .test_definition import (
    TweetyNetDefinition,
)
from .test_tweetynet import LABELMAPS, INPUT_SHAPES


# pytest.mark.parametrize vals for test_init_with_definition
MODEL_DEFS = (
    TweetyNetDefinition,
)

TEST_INIT_ARGVALS = itertools.product(LABELMAPS, INPUT_SHAPES, MODEL_DEFS)


class TestFrameClassificationModel:

    @pytest.mark.parametrize(
        'labelmap, input_shape, definition',
        TEST_INIT_ARGVALS
    )
    def test_init(self,
                  labelmap,
                  input_shape,
                  definition,
                  monkeypatch):
        """Test FrameClassificationModel.__init__ works as expected"""
        # monkeypatch a definition so we can test __init__
        definition = vak.models.definition.validate(definition)
        monkeypatch.setattr(
            vak.models.FrameClassificationModel,
            'definition',
            definition,
            raising=False
        )
        num_input_channels, num_freqbins = input_shape[0], input_shape[1]
        # network has required args that need to be determined dynamically
        network = definition.network(len(labelmap), num_input_channels, num_freqbins)
        model = vak.models.FrameClassificationModel(labelmap=labelmap, network=network)

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

    MOCK_INPUT_SHAPE = torch.Size([1, 128, 44])

    @pytest.mark.parametrize(
        'definition',
        [
            TweetyNetDefinition,
        ]
    )
    def test_from_config(self,
                         definition,
                         # our fixtures
                         specific_config,
                         # pytest fixtures
                         monkeypatch,
                         ):
        definition = vak.models.definition.validate(definition)
        model_name = definition.__name__.replace('Definition', '')
        toml_path = specific_config('train', model_name, audio_format='cbin', annot_format='notmat')
        cfg = vak.config.parse.from_toml_path(toml_path)

        # stuff we need just to be able to instantiate network
        labelmap = vak.common.labels.to_map(cfg.prep.labelset, map_unlabeled=True)

        monkeypatch.setattr(
            vak.models.FrameClassificationModel, 'definition', definition, raising=False
        )

        config = vak.config.model.config_from_toml_path(toml_path, cfg.train.model)
        num_input_channels, num_freqbins = self.MOCK_INPUT_SHAPE[0], self.MOCK_INPUT_SHAPE[1]

        config["network"].update(
            num_classes=len(labelmap),
            num_input_channels=num_input_channels,
            num_freqbins=num_freqbins
        )

        model = vak.models.FrameClassificationModel.from_config(config=config, labelmap=labelmap)
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
