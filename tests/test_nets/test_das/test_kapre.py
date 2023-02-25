import inspect

import numpy as np
import pytest
import torch

import vak.nets.das.kapre


@pytest.mark.parametrize(
    'input_length, filter_size, padding, stride, dilation, expected_output_length',
    [
        # we just test we get expected answer for DAS, not that whole function is correct
        (1024, 64, 'same', 16, 1, 64),
    ]
)
def test_conv_output_length(input_length, filter_size, padding, stride, dilation, expected_output_length):
    output_length = vak.nets.das.kapre.conv_output_length(
        input_length, filter_size, padding, stride, dilation
    )
    assert output_length == expected_output_length


@pytest.mark.parametrize(
    'n_dft',
    [
        64,
        512,
        1024,
        2048,
    ]
)
def test_get_stft_kernels(n_dft):
    real, img = vak.nets.das.kapre.get_stft_kernels(n_dft)
    for out_tensor in (real, img):
        assert isinstance(out_tensor, torch.Tensor)
        assert out_tensor.dtype == torch.float32
        assert out_tensor.shape[0] == n_dft
        assert out_tensor.shape[-1] == int(n_dft // 2 + 1)


@pytest.mark.parametrize(
    'nb_pre_dft',
    [
        64,
    ]
)
def test_get_stft_kernels_allclose(nb_pre_dft, specific_stft_kernels_factory):
    expected_real, expected_imag = specific_stft_kernels_factory(nb_pre_dft)
    real, imag = vak.nets.das.kapre.get_stft_kernels(nb_pre_dft)
    real_np, imag_np = real.cpu().numpy(), imag.cpu().numpy()
    for out, expected in zip(
            (real_np, imag_np), (expected_real, expected_imag)
    ):
        assert np.allclose(
            out, expected
        )


def test_amplitude_to_decibel_real_data(amp_to_db_in_out_tuples, device):
    # set up ----
    amp_in, expected_amp_out = amp_to_db_in_out_tuples
    amp_in = torch.from_numpy(amp_in).to(device)

    # get defaults
    sig = inspect.signature(vak.nets.das.kapre.amplitude_to_decibel)
    params = sig.parameters
    default_dynamic_range = params['dynamic_range'].default

    # actually call function
    amp_out = vak.nets.das.kapre.amplitude_to_decibel(amp_in)

    amp_out = amp_out.cpu().numpy()

    # test
    assert amp_out.max() == 0.
    assert amp_out.min() >= -default_dynamic_range  # internally we set to negative

    np.testing.assert_allclose(
        amp_out,
        expected_amp_out,
        atol=1e-5, rtol=1e-3,
    )


# from https://www.tensorflow.org/api_docs/python/tf/nn#same_padding_2:
# Note that the division by 2 means that there might be cases when the padding on both sides
# (top vs bottom, right vs left) are off by one. In this case, the bottom and right sides
# always get the one additional padded pixel.
# For example, when pad_along_height is 5, we pad 2 pixels at the top and 3 pixels at the bottom.
# Note that this is different from existing libraries such as PyTorch and Caffe,
# which explicitly specify the number of padded pixels and always pad the same number of pixels on both sides.
@pytest.mark.parametrize(
    'pad_size, expected_padding_tuple',
    [
        (5, (2, 3, 2, 3)),
        (6, (3, 3, 3, 3)),
    ]
)
def test_get_same_pad_tuple(pad_size, expected_padding_tuple):
    padding_tuple = vak.nets.das.kapre.get_same_pad_tuple(pad_h=pad_size, pad_w=pad_size)
    assert padding_tuple == expected_padding_tuple


# TODO: fix
@pytest.mark.xfail(msg='Numerical/implementation error')
def test_conv2d_same(inputs_conv2d_outputs, device, n_dft=64, nb_pre_conv=4):
    inputs, expected_conv2d_real, expected_conv2d_imag = inputs_conv2d_outputs
    inputs, expected_conv2d_real, expected_conv2d_imag = (inputs.to(device),
                                                          expected_conv2d_real.to(device),
                                                          expected_conv2d_imag.to(device))
    # permute output tensor dimensions to be in same order we use in torch
    # (batch, n_dft, 1, channels/filters) -> (batch, channels/filters, n_dft, 1)
    expected_conv2d_real = torch.permute(expected_conv2d_real, (0, 3, 1, 2))
    expected_conv2d_imag = torch.permute(expected_conv2d_imag, (0, 3, 1, 2))

    inputs = torch.permute(inputs, (0, 2, 1))  # -> channels first

    dft_real_kernels, dft_imag_kernels = vak.nets.das.kapre.get_stft_kernels(n_dft)
    # permute dimensions of dft kernels to be what pytorch conv2d uses
    # this is exact same permutation vak.nets.das.kapre.Spectrogram does
    # after calling get_stft_kernels (so we can test that function returns expected)
    dft_real_kernels = torch.permute(dft_real_kernels, (3, 2, 0, 1))
    dft_imag_kernels = torch.permute(dft_imag_kernels, (3, 2, 0, 1))

    # next line, what ``das.models`` does. No idea how one picks ``nb_pre_conv``.
    n_hop = 2 ** nb_pre_conv

    inputs = torch.unsqueeze(inputs, -1)
    subsample = (n_hop, 1)
    output_real = vak.nets.das.kapre.conv2d_same(inputs, dft_real_kernels, stride=subsample)
    output_imag = vak.nets.das.kapre.conv2d_same(inputs, dft_imag_kernels, stride=subsample)

    assert output_real.shape == torch.Size([32, 33, 64, 1])
    assert output_imag.shape == torch.Size([32, 33, 64, 1])

    np.testing.assert_allclose(
        output_real.detach().numpy(),
        expected_conv2d_real.numpy(),
        atol=1e-5, rtol=1e-3,
    )

    np.testing.assert_allclose(
        output_imag.detach().numpy(),
        expected_conv2d_imag.numpy(),
        atol=1e-5, rtol=1e-3,
    )


# TODO: fix
@pytest.mark.xfail(msg='Numerical/implementation error')
def test_spectrogram_mono_real_data(inputs_stft_out_tuples, device, n_dft=64, nb_pre_conv=4):
    inputs, stft_out_expected = inputs_stft_out_tuples
    inputs, stft_out_expected = inputs.to(device), stft_out_expected.to(device)

    inputs = torch.permute(inputs, (0, 2, 1))  # -> channels first

    dft_real_kernels, dft_imag_kernels = vak.nets.das.kapre.get_stft_kernels(n_dft)
    dft_real_kernels = torch.permute(dft_real_kernels, (3, 2, 0, 1))
    dft_imag_kernels = torch.permute(dft_imag_kernels, (3, 2, 0, 1))

    n_hop = 2 ** nb_pre_conv
    stft = vak.nets.das.kapre.Spectrogram(
        num_samples=inputs.shape[2],
        n_audio_channels=inputs.shape[1],
        n_dft=pre_nb_dft,
        n_hop=n_hop,
        return_decibel_spectrogram=True,
        power_spectrogram=1.0,
        trainable_kernel=True,
    )


    inputs = torch.unsqueeze(inputs, -1)
    subsample = (n_hop, 1)
    output_real = vak.nets.das.kapre.conv2d_same(inputs, dft_real_kernels, stride=subsample)
    output_imag = vak.nets.das.kapre.conv2d_same(inputs, dft_imag_kernels, stride=subsample)

    assert output_real.shape == torch.Size([32, 33, 64, 1])
    assert output_imag.shape == torch.Size([32, 33, 64, 1])


# TODO: fix
@pytest.mark.xfail(msg='Numerical/implementation error')
def test_spectrogram_real_data(inputs_stft_out_tuples,
                               device,
                               nb_pre_conv=4,
                               pre_nb_dft=64,
                               ):
    inputs, stft_out_expected = inputs_stft_out_tuples
    inputs, stft_out_expected = inputs.to(device), stft_out_expected.to(device)
    inputs = torch.permute(inputs, (0, 2, 1))  # -> channels first
    stft_out_expected = torch.permute(stft_out_expected, (0, 1, 3, 2))

    n_hop = 2 ** nb_pre_conv
    stft = vak.nets.das.kapre.Spectrogram(
        num_samples=inputs.shape[2],
        n_audio_channels=inputs.shape[1],
        n_dft=pre_nb_dft,
        n_hop=n_hop,
        return_decibel_spectrogram=True,
        power_spectrogram=1.0,
        trainable_kernel=True,
    )
    stft_out = stft(inputs)

    np.testing.assert_allclose(
        stft_out.detach().numpy(),
        stft_out_expected,
        atol=1e-5, rtol=1e-3,
    )
