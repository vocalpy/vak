import inspect

import torch
import pytest

import vak.nets


class TestConvEncoder:

    @pytest.mark.parametrize(
        'input_shape',
        [
            (
                    1, 128, 128,
            ),
            (
                    1, 256, 256,
            ),
        ]
    )
    def test_init(self, input_shape):
        """test we can instantiate ConvEncoder
        and it has the expected attributes"""
        net = vak.nets.ConvEncoder(input_shape)
        assert isinstance(net, vak.nets.ConvEncoder)
        for expected_attr, expected_type in (
            ('input_shape', tuple),
            ('num_input_channels', int),
            ('conv', torch.nn.Module),
            ('encoder', torch.nn.Module),
        ):
            assert hasattr(net, expected_attr)
            assert isinstance(getattr(net, expected_attr), expected_type)

        assert net.input_shape == input_shape

    @pytest.mark.parametrize(
        'input_shape, batch_size',
        [
            ((1, 128, 128,), 32),
            ((1, 256, 256,), 64),
        ]
    )
    def test_forward(self, input_shape, batch_size):
        """test we can forward a tensor through a ConvEncoder instance
        and get the expected output"""

        input = torch.rand(batch_size, *input_shape)  # a "batch"
        net = vak.nets.ConvEncoder(input_shape)
        out = net(input)
        assert isinstance(out, torch.Tensor)

