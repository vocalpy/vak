"""Deep Audio Segmenter neural network architecture"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import torch
import torch.nn.functional as F

from .kapre import Spectrogram
from .nn import ChannelNormalization, CausalConv1d, UpSampling1D


class ResidualBlock(torch.nn.Module):
    """A ``torch`` implementation of ``das.tcn.TCN.residual_block``,
    described in Figure 1 as a Residual Block (panel E, right side).

    Adapted from
    https://github.com/janclemenslab/das/blob/9ea349f13bdde7f44ba0506c5601078b2617cb45/src/das/tcn/tcn.py#L49
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()

        # ---- Make layers outside sequential so we can add these comments ----
        # in the DAS implementation there is only one conv layer;
        # TCN-Keras now uses two to match the original Bai et al. 2018 paper
        conv = CausalConv1d(n_inputs, n_outputs, kernel_size,
                            stride=stride, dilation=dilation)
        # Note that the DAS TCN implementation defaults to 0.0,
        # and all models in the paper used this
        # while Bai et al. used a dropout of 0.2.
        # Note also that the keras implementation uses SpatialDropout1d,
        # that randomly drops entire *channels*,
        # whereas Bai et al. 2018 use "vanilla" dropout that drops output from *units*.
        # The use of ``torch.nn.Dropout1d`` here matches the DAS behavior.
        dropout_ = torch.nn.Dropout1d(dropout)

        self.net = torch.nn.Sequential(
            conv,
            torch.nn.ReLU(),
            ChannelNormalization(),
            dropout_,
        )

        # this 1x1 convolution is like the parametrized skip connection in WaveNet
        self.conv_1x1 = torch.nn.Conv1d(n_inputs, n_outputs, 1)
        self.relu = torch.nn.ReLU()

        self.init_weights()

    def init_weights(self):
        # we match how ``keras`` initializes weights
        torch.nn.init.xavier_normal_(self.net[0].conv1.weight)
        torch.nn.init.xavier_normal_(self.conv_1x1.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        res = self.conv_1x1(out)
        return self.relu(x + res), out


class TCNBlock(torch.nn.Module):
    """A ``torch`` implementation of the DAS TCN Block.

    As described in Figure 1, panel E, left side.

    Adapted from
    https://github.com/janclemenslab/das/blob/9ea349f13bdde7f44ba0506c5601078b2617cb45/src/das/tcn/tcn.py#L117
    """

    def __init__(self,
                 num_inputs: int,
                 num_channels: int = 64,
                 num_blocks: int = 1,
                 kernel_size: int = 2,
                 dropout: float = 0.0,
                 stride: int | tuple = 1,
                 dilations: Sequence[int] = (1, 2, 4, 8, 16,),
                 use_skip_connections: bool = True,
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.use_skip_connections = use_skip_connections

        # Note this first layer is (1x1), has the effect of making output be (num time bins x num time bins)
        self.conv1 = CausalConv1d(in_channels=num_inputs, out_channels=num_channels,
                                  kernel_size=1, stride=1, dilation=1)

        self.tcn_layers = torch.nn.ModuleList()
        # The paper refers to the higher-level structure added in the outer loop here as a "TCN Block".
        # The keras TCN layer implementation calls them ``stacks``.
        # We use the name ``blocks`` to make correspondence between paper and code clear.
        for stack_num in range(num_blocks):
            for layer_ind, dilation in enumerate(dilations):
                self.tcn_layers.append(
                    ResidualBlock(num_channels, num_channels,
                                  kernel_size, stride=stride,
                                  dilation=dilation, dropout=dropout)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        if self.use_skip_connections:
            skip_connections = []
        else:
            skip_connections = None

        for tcn_layer in self.tcn_layers:
            x, skip_out = tcn_layer(x)
            if skip_connections:
                skip_connections.append(skip_out)

        if skip_connections:
            x = torch.add(x, *skip_connections)

        x = F.relu(x)
        return x


class DASNet(torch.nn.Module):
    """Deep Audio Segmenter network architecture,
    as described in [1]_.
    Translated to PyTorch from [2]_.
    A Temporal Convolution Network [3]_
    with an optional spectrotemporal network front end.

    References
    ----------
    .. [1] Elsa Steinfath, Adrian Palacios, Julian Rottschäfer, Deniz Yuezak, Jan Clemens (2021).
       Fast and accurate annotation of acoustic signals with deep neural networks. eLife 10:e68837.
       https://doi.org/10.7554/eLife.68837
    .. [2] https://github.com/janclemenslab/das
    .. [3] Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun.
       "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling."
       arXiv preprint arXiv:1803.01271 (2018).
    """
    def __init__(self,
                 num_samples: int,
                 num_classes: int,
                 n_audio_channels: int = 1,
                 num_filters: int = 32,
                 kernel_size: int = 32,
                 num_blocks: int = 2,
                 nb_pre_conv: int | None = None,
                 pre_nb_dft: int = 64,
                 power_spectrogram: float = 1.0,
                 return_decibel_spectrogram: bool = True,
                 trainable_kernel: bool = True,
                 use_skip_connections: bool = True,
                 dilations: tuple[int] = (1, 2, 4, 8, 16,),
                 dropout: float = 0.0,
                 ):
        r"""Initialize a new DASNet instance.

        Parameters
        ----------
        num_samples : int
            Input size of window from audio signal,
            given in number of samples, e.g., 2048.
        num_classes : int
            Number of possible classes that can be predicted for each time bin
            in the output, e.g., 10.
        n_audio_channels : int
            Number of channels in audio. Default is 1.
        num_filters : int
            Number of filters / channels used by convolutional layers
            in Temporal Convolution Network. Defaults to 16.
        kernel_size : int
            Size of kernel in convolutional layers
             in Temporal Convolution Network. Defaults to 3.
        num_blocks : int
            Number of blocks in Temporal Convolutional Network,
            i.e. repeats of the Temporal Convolutional Block,
            where each block contains the same number of residual
            blocks as there are values in ``dilations``.
            Default is 2.
        nb_pre_conv : int
            If greater than zero,
            adds a single STFT layer with a hop size of
            :math:``2^\text{nb_pre_conv}`` before the
            Temporal Convolutional Network,
            that also serves to downsample the signal.
            Useful for speeding up training
            by reducing the sample rate early in the network.
            Defaults to ``None`` (no STFT layer, no downsampling).
        pre_nb_dft: int
            Duration of filters (in samples) for the STFT frontend.
            Number of filters is pre_nb_dft // 2 + 1. Defaults to 64.
        power_spectrogram : float
            Raise the output of the ``Spectrogram``
            to this power. Default is 1.0,
            an amplitude spectrogram.
            Applied before any ``amplitude_to_decibel``
            transform.
        return_decibel_spectrogram : bool
            If True, the output of the ``Spectrogram``
            will be converted to decibels by
            ``vak.nets.das.amplitude_to_decibel``.
            Default is True.
        trainable_kernel : bool
            If True, update weights in the kernels
            used by ``vak.nets.das.Spectrogram``
            during training. Default is True.
        use_skip_connections : bool
            If True, use skip connections in the Temporal Convolutional Network.
            Default is False.
        dilations : Sequence of int
            Series of dilations used by Temporal Convolution Network.
            Default is (1, 2, 4, 8, 16).
        dropout : float
            Percent of spatial (channel-wise) dropout to apply in
            1-D causal convolutions. Default is 0.0.
        """
        if nb_pre_conv is not None:
            if not (
                nb_pre_conv > 0 and isinstance(nb_pre_conv, int)
            ):
                raise ValueError(
                    f'nb_pre_conv must be a positive integer but was: {nb_pre_conv}'
                )

        super().__init__()

        # add config params as attribs so we can inspect
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.n_audio_channels = n_audio_channels
        self.num_blocks = num_blocks
        self.nb_pre_conv = nb_pre_conv
        self.pre_nb_dft = pre_nb_dft
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.power_spectrogram = power_spectrogram
        self.return_decibel_spectrogram = return_decibel_spectrogram
        self.trainable_kernel = trainable_kernel
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.dropout = dropout

        if nb_pre_conv:
            n_hop = 2 ** nb_pre_conv
            self.stft = Spectrogram(
                    num_samples=num_samples,
                    n_audio_channels=n_audio_channels,
                    n_dft=pre_nb_dft,
                    n_hop=n_hop,
                    return_decibel_spectrogram=return_decibel_spectrogram,
                    power_spectrogram=power_spectrogram,
                    trainable_kernel=trainable_kernel,
                )
            upsample = True
            # stft output_shape is (num audio channels, num freq bins from DFT, num time bins)
            # we want num freq bins as num_inputs for TCNBlock,
            # this becomes "channels in" for TCN in the DAS model.
            # we can imagine learning to predict a sequence for each freq bin in the spectrogram
            num_inputs = self.stft.output_shape[1]
        else:
            self.stft = None
            upsample = False
            # next line, num_inputs becomes input to first conv1d layer in TCN
            # so we map (audio channels, time steps) -> (conv1d channels, time steps)
            num_inputs = n_audio_channels

        self.tcn = TCNBlock(num_inputs, num_filters, num_blocks,
                            kernel_size, dropout,
                            dilations=dilations,
                            use_skip_connections=use_skip_connections)
        FAKE_BATCH_SIZE = 8
        if nb_pre_conv:
            # ignoring the extra channel dim we get out that we squeeze in forward
            mock_input = torch.rand(FAKE_BATCH_SIZE, *self.stft.output_shape[-2:])
        else:
            mock_input = torch.rand(FAKE_BATCH_SIZE, n_audio_channels, num_inputs)  # i.e., num_samples
        mock_tcn_out = self.tcn(mock_input)
        # next two lines: we want to map from channels to class for each time bin.
        # to do this in the forward method, we permute (batch, channels, time) -> (batch, time, channels)
        # then pass through the linear layer, then permute back and upsample.
        num_channels_out = mock_tcn_out.shape[1]
        self.fc = torch.nn.Linear(in_features=num_channels_out,
                                  out_features=num_classes)

        if upsample:
            # output size should be inverse of downsampling by STFT.
            # e.g. if we downsample from 1024 to 64 time bins,
            # we need to upsample 2 ** 4 = 16 * 64 -> 1024.
            # Notice also we upsample across *last* dimension which will be time,
            # after permuting again in the forward method to get dims in correct order for loss.
            self.upsample = UpSampling1D(size=n_hop, dim=-1)
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stft:
            x = self.stft(x)
            # remove channel dimension, added by STFT that uses Conv2D
            x = torch.squeeze(x)

        x = self.tcn(x)
        # this permutation is necessary in pytorch (but not the original keras implementation of DAS)
        # because pytorch is "channels first" by default.
        x = torch.permute(x, (0, 2, 1))
        x = self.fc(x)
        # Permute again to have (batch, class, time step) as required by loss function.
        # Also, any upsample op expects to operate along the last dimension, so it should be time.
        x = torch.permute(x, (0, 2, 1))
        if self.upsample:
            x = self.upsample(x)

        return x


@dataclasses.dataclass
class DASNetConfig:
    """Dataclass that represents configuration
    for Deep Audio Segmenter

    Attributes
    ----------
    num_samples : int
        Input size of window from audio signal in number of samples, e.g. 2048.
    num_classes : int
        Number of possible classes that can be predicted for each time bin
        in the output, e.g., 10.
    num_filters : int
        Number of filters / channels used by convolutional layers
        in Temporal Convolution Network. Defaults to 16.
    kernel_size : int
        Size of kernel in convolutional layers
         in Temporal Convolution Network. Defaults to 3.
    num_blocks : int
        Number of blocks in Temporal Convolutional Network,
        i.e. repeats of the Temporal Convolutional Block,
        where each block contains the same number of residual
        blocks as there are values in ``dilations``.
        Default is 2.
    nb_pre_conv : int
        If greater than zero,
        adds a single STFT layer with a hop size of
        ``2 ** nb_pre_conv`` before the Temporal Convolutional Network,
        that also serves to downsample the signal.
        Useful for speeding up training
        by reducing the sample rate early in the network.
        Defaults to None (no downsampling).
    pre_nb_dft: int
        Duration of filters (in samples) for the STFT frontend.
        Number of filters is pre_nb_dft // 2 + 1. Defaults to 64.
    power_spectrogram : float
        Raise the output of the ``Spectrogram``
        to this power. Default is 1.0,
        an amplitude spectrogram.
        Applied before any ``amplitude_to_decibel``
        transform.
    return_decibel_spectrogram : bool
        If True, the output of the ``Spectrogram``
        will be converted to decibels by
        ``vak.nets.das.amplitude_to_decibel``.
        Default is True.
    trainable_kernel : bool
        If True, update weights in the kernels
        used by ``vak.nets.das.Spectrogram``
        during training. Default is True.
    use_skip_connections : bool
        If True, use skip connections in the Temporal Convolutional Network.
        Default is False.
    dilations : Sequence of int
        Series of dilations used by Temporal Convolution Network.
        Default is (1, 2, 4, 8, 16).
    dropout : float
        Percent of spatial (channel-wise) dropout to apply in
        1-D causal convolutions. Default is 0.0.
    """
    num_samples: int
    num_classes: int
    num_blocks: int
    nb_pre_conv: int | None = None
    pre_nb_dft: int = 64
    n_audio_channels: int = 1
    num_filters: int = 32
    kernel_size: int = 32
    power_spectrogram: float = 1.0
    return_decibel_spectrogram: bool = True
    trainable_kernel: bool = True
    use_skip_connections: bool = True
    dilations: tuple[int] = (1, 2, 4, 8, 16)
    dropout: float = 0.0


DASNetFlySinglechannelConfig = DASNetConfig(
    num_samples=4096,
    num_classes=3,
    num_filters=32,
    kernel_size=32,
    num_blocks=3,
    power_spectrogram=1.0,
    return_decibel_spectrogram=True,
)


def dasnet_fly_singlechannel():
    return DASNet(**dataclasses.asdict(DASNetFlySinglechannelConfig))


DASNetFlyMultichannelConfig = DASNetConfig(
    num_samples=2048,
    n_audio_channels=9,
    num_classes=2,
    num_filters=32,
    kernel_size=32,
    num_blocks=4,
    power_spectrogram=1.0,
    return_decibel_spectrogram=True,
)

def dasnet_fly_multichannel():
    return DASNet(**dataclasses.asdict(DASNetFlyMultichannelConfig))


DASNetMouseConfig = DASNetConfig(
    num_samples=8192,
    num_classes=2,
    num_filters=32,
    kernel_size=16,
    nb_pre_conv=4,
    num_blocks=2,
    power_spectrogram=1.0,
    return_decibel_spectrogram=True,
)


def dasnet_mouse():
    return DASNet(**dataclasses.asdict(DASNetMouseConfig))


DASNetMarmosetConfig = DASNetConfig(
    num_samples=8192,
    num_classes=5,
    num_filters=32,
    kernel_size=16,
    nb_pre_conv=4,
    num_blocks=2,
    power_spectrogram=1.0,
    return_decibel_spectrogram=True,
)


def dasnet_marmoset():
    return DASNet(**dataclasses.asdict(DASNetMarmosetConfig))


DASNetBengaleseFinchConfig = DASNetConfig(
    num_samples=1024,
    num_classes=49,
    num_filters=64,
    kernel_size=32,
    nb_pre_conv=4,
    num_blocks=4,
    power_spectrogram=1.0,
    return_decibel_spectrogram=True,
)


def dasnet_bengalese_finch():
    return DASNet(**dataclasses.asdict(DASNetBengaleseFinchConfig))


DASNetZebraFinchConfig = DASNetConfig(
    num_samples=2048,
    num_classes=7,
    num_filters=64,
    kernel_size=32,
    nb_pre_conv=4,
    num_blocks=4,
    power_spectrogram=1.0,
    return_decibel_spectrogram=True,
)


def dasnet_zebra_finch():
    return DASNet(**dataclasses.asdict(DASNetZebraFinchConfig))
