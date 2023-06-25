"""Neural network operations and layers used by DASNet blocks.
Written as torch modules."""
from __future__ import annotations

import torch


class ChannelNormalization(torch.nn.Module):
    """Normalize a tensor along the channel dimension,
    dividing all values in each channel by the maximum value
    in that channel.

    Attributes
    ----------
    dim : int
        Dimension. Default is 1,
        the channel dimension,
        assuming a tensor with dimensions
        (batch, channel, time step).
    keepdim : bool
        If True, do not collapse dimension
        when computing the max function.
        Default is True.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    x_normalized : torch.Tensor

    Notes
    -----
    Adapted from:
    https://github.com/janclemenslab/das/blob/9ea349f13bdde7f44ba0506c5601078b2617cb45/src/das/tcn/tcn.py#L15
    """

    def __init__(self, dim: int = 1, keepdim: bool = True):
        """Initialize a new ChannelNormalization instance.

        Parameters
        ----------
        dim : int
            Dimension. Default is 1,
            the channel dimension,
            assuming a tensor with dimensions
            (batch, channel, time step).
        keepdim : bool
            If True, do not collapse dimension
            when computing the max function.
            Default is True.
        """
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor along the channel dimension,
        dividing all values in each channel by the maximum value
        in that channel.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        x_normalized : torch.Tensor
        """
        # Note the original implementation uses 1e-5 to avoid divide-by-zero.
        # This might seem like a pretty large value for epsilon, but we use it to replicate the original.
        x_per_channel_max = torch.max(torch.abs(x), dim=self.dim, keepdim=self.keepdim)[0] + 1e-5  # [0] for values
        return x / x_per_channel_max


class CausalConv1d(torch.nn.Module):
    """Causal 1-d Convolution.

    Attributes
    ----------
    in_channels: int
        Number of channels in the input.
    out_channels: int
        Number of channels produced by the convolution.
    kernel_size: int, tuple
        Size of the convolving kernel,
    stride: int or tuple, optional
        Stride of the convolution. Default: 1.
    dilation: int or tuple, optional
        Spacing between kernel elements. Default: 1.
    groups: int, optional
        Number of blocked connections from input
        channels to output channels. Default: 1.
    bias: bool, optional
        If ``True``, adds a learnable bias to the output.
        Default: ``True``axis.

    Notes
    -----
    Note there is no padding parameter,
    since we determine how to left-pad the input
    to produce causal convolutions
    according to the dilation value and kernel size.

    This implements causal padding as in `keras`,
    since this is what is used by the TCN implementation:
    https://github.com/janclemenslab/das/blob/9ea349f13bdde7f44ba0506c5601078b2617cb45/src/das/tcn/tcn.py#L56
    See: https://theblog.github.io/post/convolution-in-autoregressive-neural-networks/
    The original Bai et al. 2018 paper instead pads both sides,
    then removes extra padding on the right with a `Chomp1D` operation.
    The left padding is slightly more efficient.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 device: str | None = None,
                 dtype: str | None = None):
        super().__init__()
        self.leftpad = (kernel_size - 1) * dilation
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                                     stride=stride, padding='valid', dilation=dilation,
                                     groups=groups, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a causal convolution.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        out : torch.Tensor
            Output of convolution operation.
        """
        x = torch.nn.functional.pad(x, (self.leftpad, 0))  # left pad last dimension, time
        return self.conv1(x)


def upsampling1d(x: torch.Tensor, size: int, dim=1) -> torch.Tensor:
    """Functional form of an Upsampling1d layer
    that works in the same way as ``keras.layers.Upsampling1D``,
    by simply repeating elements,
    instead of using an interpolation algorithm.

    For ``keras.layers`` implementation, see:
    https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/layers/reshaping/up_sampling1d.py#L29

    Parameters
    ----------
    x : torch.Tensor
    size : int

    Returns
    -------
    x_upsampled
    """
    return torch.repeat_interleave(x, repeats=size, dim=dim)


class UpSampling1D(torch.nn.Module):
    def __init__(self, size: int, dim: int = 1):
        super().__init__()
        self.size = size
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upsampling1d(x, self.size, self.dim)
