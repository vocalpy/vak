import inspect

import pytest
import torch

import vak.models


class ConvEncoderUMAPDefinition:
    network = {"encoder": vak.nets.ConvEncoder}
    loss = vak.nn.UmapLoss
    optimizer = torch.optim.AdamW
    metrics = {
        "acc": vak.metrics.Accuracy,
        "levenshtein": vak.metrics.Levenshtein,
        "segment_error_rate": vak.metrics.SegmentErrorRate,
        "loss": torch.nn.CrossEntropyLoss,
    }
    default_config = {
        "optimizer": {"lr": 1e-3},
    }


class TestParametricUMAPModel:

    @pytest.mark.parametrize(
        'input_shape, definition',
        [
            ((1, 128, 128), ConvEncoderUMAPDefinition),
        ]
    )
    def test_init(
            self,
            input_shape,
            definition,
            monkeypatch,
    ):
        """Test ParametricUMAPModel.__init__ works as expected"""
        # monkeypatch a definition so we can test __init__
        definition = vak.models.definition.validate(definition)
        monkeypatch.setattr(
            vak.models.ParametricUMAPModel,
            'definition',
            definition,
            raising=False
        )
        network = {'encoder': vak.nets.ConvEncoder(input_shape)}
        model = vak.models.ParametricUMAPModel(network=network)

        # now test that attributes are what we expect
        assert isinstance(model, vak.models.ParametricUMAPModel)
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

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        'input_shape, definition',
        [
            ((1, 128, 128), ConvEncoderUMAPDefinition),
        ]
    )
    def test_from_config(
            self,
            input_shape,
            definition,
            specific_config,
            monkeypatch,
    ):
        definition = vak.models.definition.validate(definition)
        model_name = definition.__name__.replace('Definition', '')
        toml_path = specific_config('train', model_name, audio_format='cbin', annot_format='notmat')
        cfg = vak.config.parse.from_toml_path(toml_path)

        monkeypatch.setattr(
            vak.models.ParametricUMAPModel, 'definition', definition, raising=False
        )

        config = vak.config.model.config_from_toml_path(toml_path, cfg.train.model)
        config["network"].update(
            encoder=dict(input_shape=input_shape)
        )

        model = vak.models.ParametricUMAPModel.from_config(config=config)
        assert isinstance(model, vak.models.ParametricUMAPModel)

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
