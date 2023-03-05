import dataclasses

import pytest
import pytorch_lightning as lightning
import torch

import vak.models


class TestDAS:
    def test_model_is_decorated(self):
        assert issubclass(vak.models.DAS,
                          vak.models.WindowedFrameClassificationModel)
        assert issubclass(vak.models.DAS,
                          vak.models.base.Model)
        assert issubclass(vak.models.DAS,
                          lightning.LightningModule)

    @pytest.mark.parametrize(
        'net_config_dataclass',
        [
            vak.nets.das.net.DASNetBengaleseFinchConfig,
            vak.nets.das.net.DASNetFlyMultichannelConfig,
            vak.nets.das.net.DASNetFlySinglechannelConfig,
            vak.nets.das.net.DASNetMarmosetConfig,
            vak.nets.das.net.DASNetMouseConfig,
        ]
    )
    def test_init(self, net_config_dataclass):
        net_config_dict = dataclasses.asdict(net_config_dataclass)
        network = vak.nets.das.net.DASNet(**net_config_dict)
        mock_labelmap = {lbl: str(lbl) for lbl in range(net_config_dict['num_classes'])}

        model = vak.models.DAS(labelmap=mock_labelmap, network=network)

        assert isinstance(model, vak.models.DAS)
        for attr in ('network', 'loss', 'optimizer'):
            assert hasattr(model, attr)
            assert isinstance(getattr(model, attr),
                              getattr(vak.models.das.DAS.definition, attr))
        assert hasattr(model, 'metrics')
        assert isinstance(model.metrics, dict)
        for metric_name, metric_callable in model.metrics.items():
            assert isinstance(metric_callable,
                              vak.models.das.DAS.definition.metrics[metric_name])


    @pytest.mark.parametrize(
        'net_config_dataclass',
        [
            vak.nets.das.net.DASNetBengaleseFinchConfig,
            vak.nets.das.net.DASNetFlyMultichannelConfig,
            vak.nets.das.net.DASNetFlySinglechannelConfig,
            vak.nets.das.net.DASNetMarmosetConfig,
            vak.nets.das.net.DASNetMouseConfig,
        ]
    )
    def test_forward(self, net_config_dataclass):
        net_config_dict = dataclasses.asdict(net_config_dataclass)
        network = vak.nets.das.net.DASNet(**net_config_dict)
        mock_labelmap = {lbl: str(lbl) for lbl in range(net_config_dict['num_classes'])}

        model = vak.models.DAS(labelmap=mock_labelmap, network=network)

        FAKE_BATCH_SIZE = 8
        fake_input_shape = (FAKE_BATCH_SIZE,
                            net_config_dict['n_audio_channels'],
                            net_config_dict['num_samples'])
        fake_input = torch.rand(*fake_input_shape)

        out = model(fake_input)

        out_shape = out.shape
        assert out_shape[0] == FAKE_BATCH_SIZE
        assert out_shape[1] == net_config_dict['num_classes']
        assert out_shape[2] == net_config_dict['num_samples']
