import numpy as np
import pytest
import torch

import vak.nets.das


class TestChannelNormalization:

    @pytest.mark.parametrize(
        'dim, keepdim',
        [
            (None, None),
            (1, True)
        ]
    )
    def test_init(self, dim, keepdim):
        if dim and keepdim:
            channel_norm = vak.nets.das.nn.ChannelNormalization(dim=dim, keepdim=keepdim)
        else:
            # test default args
            channel_norm = vak.nets.das.nn.ChannelNormalization()

        assert isinstance(channel_norm, vak.nets.das.nn.ChannelNormalization)

        if dim and keepdim:
            assert channel_norm.dim == dim
            assert channel_norm.keepdim == keepdim

    def test_forward(self):
        # note only testing default behavior here
        channel_norm = vak.nets.das.nn.ChannelNormalization()

        x = torch.rand(10, 64, 1048)

        out = channel_norm(x)

        torch.testing.assert_allclose(
            out,
            x / (torch.max(torch.abs(x), dim=channel_norm.dim, keepdim=channel_norm.keepdim)[0] + 1e-5),
        )


class TestCausalConv1d:

    @pytest.mark.parametrize(
        'in_channels, out_channels, kernel_size, stride, dilation, use_separable',
        [
            (1024, 64, 2, 1, 1, False),
            (64, 64, 2, 1, 2, False),
            (64, 64, 2, 1, 4, False),
            (64, 64, 2, 1, 8, False),
            (64, 64, 2, 1, 16, False),
            (1024, 64, 2, 1, 1, True),
            (64, 64, 2, 1, 2, True),
            (64, 64, 2, 1, 4, True),
            (64, 64, 2, 1, 8, True),
            (64, 64, 2, 1, 16, True),
        ]
    )
    def test_init(self, in_channels, out_channels, kernel_size, stride, dilation, use_separable):
        conv1d = vak.nets.das.nn.CausalConv1d(in_channels, out_channels,
                                              kernel_size, stride=stride,
                                              dilation=dilation,
                                              use_separable=use_separable)
        assert isinstance(conv1d, vak.nets.das.nn.CausalConv1d)

    @pytest.mark.parametrize(
        'in_channels, out_channels, L, kernel_size, stride, dilation, use_separable',
        [
            (1, 64, 1024, 2, 1, 1, False),
            (64, 64, 64, 2, 1, 2, False),
            (64, 64, 64, 2, 1, 4, False),
            (64, 64, 64, 2, 1, 8, False),
            (64, 64, 64, 2, 1, 16, False),
            (1, 64, 1024, 2, 1, 1, True),
            (64, 64, 64, 2, 1, 2, True),
            (64, 64, 64, 2, 1, 4, True),
            (64, 64, 64, 2, 1, 8, True),
            (64, 64, 64, 2, 1, 16, True),
        ]
    )
    def test_forward(self, in_channels, out_channels, L, kernel_size, stride, dilation, use_separable):
        conv1d = vak.nets.das.nn.CausalConv1d(in_channels, out_channels,
                                              kernel_size, stride=stride,
                                              dilation=dilation,
                                              use_separable=use_separable)


        x = torch.rand(10, in_channels, L)
        out = conv1d(x)

        assert isinstance(x, torch.Tensor)
        # assert out.shape == x.shape


INPUT_SHAPE = (2, 2, 3)
X_NP = np.arange(np.prod(INPUT_SHAPE)).reshape(INPUT_SHAPE)
REPEATS = [2, 3, 4]
DIM = 1

@pytest.mark.parametrize(
    'repeats',
    REPEATS
)
def test_upsampling1d(repeats):
    x = torch.tensor(X_NP)
    out = vak.nets.das.nn.upsampling1d(x, size=repeats)  # has param dim, defaults to 1
    expected_x = torch.tensor(
        np.repeat(X_NP, repeats, axis=DIM)
    )
    assert torch.all(torch.eq(out, expected_x))


@pytest.mark.parametrize(
    'repeats',
    REPEATS
)
def test_Upsampling1d(repeats):
    x = torch.tensor(X_NP)

    upsample = vak.nets.das.nn.UpSampling1D(size=repeats)
    out = upsample(x)  # has param dim, defaults to 1
    expected_x = torch.tensor(
        np.repeat(X_NP, repeats, axis=DIM)
    )
    assert torch.all(torch.eq(out, expected_x))
