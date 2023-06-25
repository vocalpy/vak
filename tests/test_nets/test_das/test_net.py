import dataclasses

import pytest
import torch

import vak.nets.das


class TestResidualBlock:

    @pytest.mark.parametrize(
        'in_channels, out_channels, kernel_size, stride, dilation, dropout',
        [
            (1024, 64, 2, 1, 1, 0.0),
            (64, 64, 2, 1, 2, 0.0),
            (64, 64, 2, 1, 4, 0.0),
            (64, 64, 2, 1, 8, 0.0),
            (64, 64, 2, 1, 16, 0.0),
        ]
    )
    def test_init(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        residual_block = vak.nets.das.net.ResidualBlock(in_channels, out_channels,
                                                        kernel_size, stride=stride,
                                                        dilation=dilation, dropout=dropout)
        assert isinstance(residual_block, vak.nets.das.net.ResidualBlock)

    @pytest.mark.parametrize(
        'in_channels, out_channels, L, kernel_size, stride, dilation, dropout',
        [
            (1, 64, 1024, 2, 1, 1, 0.0),
            (64, 64, 64, 2, 1, 2, 0.0),
            (64, 64, 64, 2, 1, 4, 0.0),
            (64, 64, 64, 2, 1, 8, 0.0),
            (64, 64, 64, 2, 1, 16, 0.0),
            (1, 64, 1024, 2, 1, 1, 0.0),
        ]
    )
    def test_forward(self, in_channels, out_channels, L, kernel_size, stride, dilation, dropout):
        residual_block = vak.nets.das.net.ResidualBlock(in_channels, out_channels,
                                                        kernel_size, stride=stride,
                                                        dilation=dilation, dropout=dropout)

        if in_channels != out_channels:
            pytest.xfail(
                reason=('DAS implementation of ResidualBlock does not correctly downsample input, '
                        'see comment in module. So xfail if in_channels != out_channels')
            )
        else:
            x = torch.rand(10, in_channels, L)
            out = residual_block(x)
            assert isinstance(x, torch.Tensor)


class TestTCNBLock:
    @pytest.mark.parametrize(
        'num_inputs, num_blocks, use_skip_connections',
        [
            (64, 2, True),
            (64, 3, True),
            (64, 4, True),
        ]
    )
    def test_init(self, num_inputs, num_blocks, use_skip_connections):
        tcn = vak.nets.das.net.TCNBlock(num_inputs=num_inputs,
                                        num_blocks=num_blocks,
                                        use_skip_connections=use_skip_connections)
        assert isinstance(tcn, vak.nets.das.net.TCNBlock)
        assert len(tcn.tcn_layers) == len(tcn.dilations) * tcn.num_blocks


class TestDASNet:
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
        net = vak.nets.das.net.DASNet(**net_config_dict)

        assert isinstance(net, vak.nets.das.net.DASNet)

        for param_name, param_val in net_config_dict.items():
            assert getattr(net, param_name) == param_val

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
        net = vak.nets.das.net.DASNet(**net_config_dict)

        FAKE_BATCH_SIZE = 8
        fake_input_shape = (FAKE_BATCH_SIZE,
                            net_config_dict['n_audio_channels'],
                            net_config_dict['num_samples'])
        fake_input = torch.rand(*fake_input_shape)
        out = net(fake_input)
        out_shape = out.shape
        assert out_shape[0] == FAKE_BATCH_SIZE
        assert out_shape[1] == net_config_dict['num_classes']
        assert out_shape[2] == net_config_dict['num_samples']


@pytest.mark.parametrize(
    'net_builder_func, net_config_dataclass',
    [
        (vak.nets.das.dasnet_bengalese_finch, vak.nets.das.net.DASNetBengaleseFinchConfig),
        (vak.nets.das.dasnet_fly_multichannel, vak.nets.das.net.DASNetFlyMultichannelConfig),
        (vak.nets.das.dasnet_fly_singlechannel, vak.nets.das.net.DASNetFlySinglechannelConfig),
        (vak.nets.das.dasnet_marmoset, vak.nets.das.net.DASNetMarmosetConfig),
        (vak.nets.das.dasnet_mouse, vak.nets.das.net.DASNetMouseConfig),

    ]
)
def test_net_builders(net_builder_func, net_config_dataclass):
    if 'fly' in net_builder_func.__name__:
        pytest.xfail(
            reason='error with groups due to separableconv'
        )
    net = net_builder_func()
    net_config_dict = dataclasses.asdict(net_config_dataclass)
    for param_name, param_val in net_config_dict.items():
        assert getattr(net, param_name) == param_val
