"""A PyTorch implementation of the Kapre
functions used in DAS.

Kapre is a library of Keras Audio Preprocessors, released under MIT License.
https://github.com/keunwoochoi/kapre

Adapted from https://github.com/janclemenslab/das/tree/master/src/das/kapre.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F


def conv_output_length(input_length: int, filter_size: int, padding: str,
                       stride: int, dilation: int = 1) -> int | None:
    """Determines output length of a convolution given input length.

    Used by ``vak.nets.das.Spectrogram`` class to determine
    number of frames / time steps in output of STFT operation.

    Parameters
    ----------
    input_length: int
        Number of samples in audio input.
    filter_size: int
        Size of filter/kernel in convolutions.
    padding: str
        Type of padding.
        One of {"same", "valid", "full", "causal"}.
    stride: int
        Size of stride.
    dilation: int
        Dilation rate, default is 1.

    Returns
    -------
    output_length : int

    Notes
    -----
    Adapted from https://github.com/janclemenslab/das/tree/master/src/das/kapre.
    """
    if input_length is None:
        return None

    padding_types = {"same", "valid", "full", "causal"}
    if padding not in padding_types:
        raise ValueError(
            f"Value for `padding` not recognized: {padding}."
            f"Valid padding types are: {padding_types}"
        )

    dilated_filter_size = (filter_size - 1) * dilation + 1

    if padding == "same":
        output_length = input_length
    elif padding == "valid":
        output_length = input_length - dilated_filter_size + 1
    elif padding == "causal":
        output_length = input_length
    elif padding == "full":
        output_length = input_length + dilated_filter_size - 1

    return (output_length + stride - 1) // stride


def get_stft_kernels(n_dft: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DFT kernels for real and imaginary parts of spectrogram,
     assuming the input is real.

    Parameters
    ----------
    n_dft : int
        Number of DFT components.
        Must be a power of 2.

    Returns
    -------
    dft_real_kernels : torch.Tensor
        Kernels used for real component of signal.
        With shape (n_dft, 1, 1, nb_filter),
        where nb_filter = n_dft/2 + 1.
        The first two dimensions are the dimensions
        of the (2D) convolutional filter,
        the third dimension is a "channel".
    dft_imag_kernels : torch.Tensor
        With shape (n_dft, 1, 1, nb_filter).

    Notes
    -----
    An asymmetric Hann window (``scipy.signal.hann``)
    is applied to the kernels.
    Adapted from
    https://github.com/janclemenslab/das/blob/9ea349f13bdde7f44ba0506c5601078b2617cb45/src/das/kapre/backend.py#L42.
    This function maintains the output shape of the original function,
    that is then transposed by ``vak.nets.das.Spectrogram``.
    """
    if not (
        n_dft > 1 and ((n_dft & (n_dft - 1)) == 0)
    ):
        raise ValueError(
            f'n_dft should be > 1 and a power of 2 but was: {n_dft}'
        )

    nb_filter = int(n_dft // 2 + 1)

    # prepare DFT filters
    timesteps = np.array(range(n_dft))
    w_ks = np.arange(nb_filter) * 2 * np.pi / float(n_dft)
    dft_real_kernels = np.cos(w_ks.reshape(-1, 1) * timesteps.reshape(1, -1))
    dft_imag_kernels = -np.sin(w_ks.reshape(-1, 1) * timesteps.reshape(1, -1))

    # windowing DFT filters
    dft_window = scipy.signal.get_window("hann", n_dft, fftbins=True)
    dft_window = dft_window
    dft_window = dft_window.reshape((1, -1))
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
    dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)

    dft_real_kernels = dft_real_kernels.transpose()
    dft_imag_kernels = dft_imag_kernels.transpose()
    dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
    dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]

    dft_real_kernels = torch.from_numpy(dft_real_kernels).to(torch.float32)
    dft_imag_kernels = torch.from_numpy(dft_imag_kernels).to(torch.float32)
    return dft_real_kernels, dft_imag_kernels


def amplitude_to_decibel(x: torch.Tensor, amin: float = 1e-10,
                         dynamic_range: float = 80.0) -> torch.Tensor:
    """Convert amplitude spectrogram to decibel.

    Parameters
    ----------
    x : torch.Tensor
        Amplitude spectrogram.
    amin : float, optional
        Minimum amplitude. Smaller values are clipped to this.
        Defaults to 1e-10 (dB).
    dynamic_range : float, optional
        Dynamic range. Defaults to 80.0 (dB).

    Returns
    -------
    x_db: torch.Tensor
        With the values converted to decibels.

    Notes
    -----
    Adapted from https://github.com/janclemenslab/das/tree/master/src/das/kapre.
    """
    # I am not sure this is technically equivalent to a true amplitude -> dB conversion.
    # The STFT class by default generates power spectrograms, although the DAS network
    # then sets the default to 1.0, which means the input to this will be in amplitude.
    # But we still convert to a log scale, "linearizing" the range the rest of the network sees.
    # For literal amplitude to dB I would write this in a simpler way.
    # But it does do the same thing as the function in DAS, converted from keras to torch.
    amin = torch.tensor(amin).to(x.dtype)
    log_spec = (
            10 * torch.log(torch.maximum(x, amin)) / np.log(10)
    ).to(x.dtype)
    if x.dim() > 1:
        axis = tuple(range(x.dim())[1:])
    else:
        axis = None

    log_spec = log_spec - torch.amax(log_spec, dim=axis, keepdim=True)  # [-?, 0]
    dynamic_range = torch.tensor(-1 * dynamic_range).to(x.dtype)
    log_spec = torch.maximum(log_spec, dynamic_range)  # [-80, 0]
    return log_spec


# taken from https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351
# note that TweetyNet also has a ``Conv2d`` but reading those issues again,
# I think the TweetyNet implementation is incorrect,
# because it does not capture the TensorFlow behavior of rounding
# https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
# https://gist.github.com/Yangqing/47772de7eb3d5dbbff50ffb0d7a98964
def calc_same_pad(input_size: int, kernel_size: int, stride: int, dilation: int = 1) -> int:
    # return either calculated padding, or 0 if padding < 0
    return max(
        (math.ceil(input_size / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size,
        0
    )


def get_same_pad_tuple(pad_h:int, pad_w: int) -> tuple[int, int, int, int]:
    """Function that computes padding
    the same way as tensorflow's 'SAME', see
    https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    https://gist.github.com/Yangqing/47772de7eb3d5dbbff50ffb0d7a98964

    Parameters
    ----------
    pad_h : int
        Padding for height dimension.
    pad_w : int
        Padding for width dimension.

    Returns
    -------
    pad_left, pad_right, pad_top, pad_bottom
    """
    return (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)


def conv2d_same(input: torch.Tensor, filter: torch.Tensor,
                stride: tuple[int, int],
                dilation: tuple[int, int] = (1, 1)
                ) -> torch.Tensor:
    """functional form of ``conv2d`` that replicates
    Tensorflow's 'same' padding.

    See:
    https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    https://gist.github.com/Yangqing/47772de7eb3d5dbbff50ffb0d7a98964
    """
    ih, iw = input.size()[-2:]

    pad_h = calc_same_pad(input_size=ih, kernel_size=filter.shape[2], stride=stride[0], dilation=dilation[0])
    pad_w = calc_same_pad(input_size=iw, kernel_size=filter.shape[3], stride=stride[1], dilation=dilation[1])

    padding_tuple = get_same_pad_tuple(pad_h, pad_w)  # (pad_left, pad_right, pad_top, pad_bottom)

    if pad_h > 0 or pad_w > 0:
        input = F.pad(
            input, padding_tuple
        )
    return F.conv2d(
        input,
        filter,
        stride=stride,
        padding='valid',  # because we already pad above
    )


class Spectrogram(torch.nn.Module):
    """Torch module that computes spectrograms.

    Attributes
    ----------
    num_samples : int
        Input size of window from audio signal,
        given in number of samples, e.g., 2048.
    n_audio_channels : int
        Number of channels in audio. Default is 1.
    n_dft : int, optional
        The number of DFT points. Best if power of 2.
        Defaults to 512.
    n_hop : int, optional
        Hop length between frames in sample. Best if <= `n_dft`.
        Defaults to None.
    padding : str, optional
        Pads signal boundaries (`same` or `valid`).
        Defaults to 'same'.
    power_spectrogram : float, optional
        `2.0` for power, `1.0` for amplitude spectrogram.
        Defaults to 2.0 (power).
    return_decibel_spectrogram (bool, optional):
        Convert spectrogram values to dB.
        Defaults to False.
    trainable_kernel (bool, optional):
        If True, kernels will be optimized during training.
        Defaults to False.

    Notes
    -----
    Adapted from https://github.com/janclemenslab/das/tree/master/src/das/kapre.
    I also consulted nnAudio to help translate to PyTorch:
    https://github.com/KinWaiCheuk/nnAudio/blob/master/Installation/nnAudio/features/stft.py
    """
    def __init__(self,
                 num_samples: int,
                 n_audio_channels: int = 1,
                 n_dft: int = 512,
                 n_hop: Optional[int] = None,
                 power_spectrogram: float = 2.0,
                 return_decibel_spectrogram: bool = False,
                 trainable_kernel: bool = False):
        """Initialize a spectrogram module.

        Parameters
        ----------
        num_samples : int
            Input size of window from audio signal,
            given in number of samples, e.g., 2048.
        n_audio_channels : int
            Number of channels in audio. Default is 1.
        n_dft : int, optional
            The number of DFT points. Best if power of 2.
            Defaults to 512.
        n_hop : int, optional
            Hop length between frames in sample. Best if <= `n_dft`.
            Defaults to None.
        power_spectrogram : float, optional
            `2.0` for power, `1.0` for amplitude spectrogram.
            Defaults to 2.0 (power).
        return_decibel_spectrogram (bool, optional):
            Convert spectrogram values to dB.
            Defaults to False.
        trainable_kernel (bool, optional):
            If True, kernels will be optimized during training.
            Defaults to False.
        """
        if not (
                n_dft > 1 and ((n_dft & (n_dft - 1)) == 0)
        ):
            raise ValueError(
                f'n_dft should be > 1 and a power of 2 but was: {n_dft}'
            )
        if n_hop is None:
            n_hop = n_dft // 2

        super().__init__()

        self.n_dft = n_dft
        self.n_filter = n_dft // 2 + 1
        self.trainable_kernel = trainable_kernel
        self.n_hop = n_hop
        self.power_spectrogram = float(power_spectrogram)
        self.return_decibel_spectrogram = return_decibel_spectrogram

        self.num_samples = num_samples
        self.n_audio_channels = n_audio_channels
        self.is_mono = self.n_audio_channels == 1
        self.ch_axis_idx = 1

        # next line, "magic string" 'same' because we always use same padding, to replicate DAS behavior
        self.n_frame = conv_output_length(self.num_samples, self.n_dft, 'same', self.n_hop)

        dft_real_kernels, dft_imag_kernels = get_stft_kernels(self.n_dft)
        # we preserve the output shape of the ``das`` function for replication purposes
        # (so we can test we get the same answer with scipy + torch instead of librosa + keras)
        # then permute the dims here so the Conv2D works correctly in pytorch
        # [filter_height, filter_width, in_channels, out_channels] ->
        # (out channels, in channels, kH, kW)
        dft_real_kernels = torch.permute(dft_real_kernels, (3, 2, 0, 1))
        dft_imag_kernels = torch.permute(dft_imag_kernels, (3, 2, 0, 1))

        if trainable_kernel:
            dft_real_kernels = torch.nn.Parameter(dft_real_kernels)
            dft_imag_kernels = torch.nn.Parameter(dft_imag_kernels)
            self.register_parameter("dft_real_kernels", dft_real_kernels)
            self.register_parameter("dft_imag_kernels", dft_imag_kernels)
        else:
            self.register_buffer("dft_real_kernels", dft_real_kernels)
            self.register_buffer("dft_imag_kernels", dft_imag_kernels)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self.n_audio_channels, self.n_filter, self.n_frame

    def _spectrogram_mono(self, x: torch.Tensor) -> torch.Tensor:
        """x.shape : (None, 1, len_src),
        returns 2D batch of a mono power-spectrogram"""
        # TODO: input will actually be (batch, channel, time). Fix how?
        # add a second dummy channel dimension to use Conv2D the way Kapre does
        x = torch.unsqueeze(x, -1)
        subsample = (self.n_hop, 1)
        output_real = conv2d_same(x, self.dft_real_kernels, stride=subsample)
        output_imag = conv2d_same(x, self.dft_imag_kernels, stride=subsample)
        output = output_real ** 2 + output_imag ** 2
        # now shape is (batch size, freq, n_frame, 1) where 1 is the "color channel".
        # We permute -> batch size, 1, freq, n_frame as if it were an "image" with one "color channel"
        output = torch.permute(output, [0, 3, 1, 2])
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._spectrogram_mono(x[:, 0:1, :])
        if self.is_mono is False:
            for ch_idx in range(1, self.n_ch):
                output = torch.concat((output, self._spectrogram_mono(x[:, ch_idx, :])),
                                      dim=self.ch_axis_idx)
        # output = output[..., 0]
        if self.power_spectrogram != 2.0:
            output = torch.pow(
                # sqrt "undoes" power spectrogram, then we raise to other power
                torch.sqrt(output), self.power_spectrogram
            )
        if self.return_decibel_spectrogram:
            output = amplitude_to_decibel(output)
        return output
