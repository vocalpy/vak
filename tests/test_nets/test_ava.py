import torch
import pytest

import vak.nets


class TestAVA:

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
        """test we can instantiate AVA
        and it has the expected attributes"""
        net = vak.nets.AVA(input_shape)
        assert isinstance(net, vak.nets.AVA)
        for expected_attr, expected_type in (
            ('input_shape', tuple),
            ('in_channels', int),
            ('x_shape', tuple),
            ('x_dim', int),
            ('encoder', torch.nn.Module),
            ('shared_encoder_fc', torch.nn.Module),
            ('mu_fc', torch.nn.Module),
            ('cov_factor_fc', torch.nn.Module),
            ('cov_diag_fc', torch.nn.Module),
            ('decoder_fc', torch.nn.Module),
            ('decoder', torch.nn.Module),
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
        net = vak.nets.AVA(input_shape)
        out = net(input)
        assert len(out) == 3
        x_rec, z, latent_dist = out
        for tensor in (x_rec, z):
            assert isinstance(tensor, torch.Tensor)
        assert isinstance(latent_dist, torch.distributions.LowRankMultivariateNormal)
