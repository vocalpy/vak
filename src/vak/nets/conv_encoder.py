from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """Convolutional encoder,
    used by Parametric UMAP model.
    """

    def __init__(
        self,
        input_shape: tuple[int],
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        conv_padding: int = 1,
        n_features_linear: int = 256,
        n_components: int = 2,
    ):
        """Initialize a ConvEncoder instance.

        Parameters
        ----------
        input_shape : tuple
            with 3 elements corresponding to dimensions of spectrogram:
            (channels, frequency bins, time bins).
            I.e., we assume input is a spectrogram and treat it like an image,
            typically with one channel; the rows are frequency bins,
            and the columns are time bins.
        conv1_filters : int
            Number of filters in first convolutional layer. Default is 64.
        conv2_filters : int
            Number of filters in second convolutional layer. Default is 128.
        conv_kernel_size : tuple
            Size of kernels, i.e. filters, in convolutional layers. Default is 3.
        conv_padding : int
            Amount of padding for convolutional layers. Default is 1.
        n_components : int
            Number of components of latent space that encoder maps to. Default is 2.
        """
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError(
                "Expected input_shape with length 3, (channels, height, width), "
                f"but input shape was length {len(input_shape)}. "
                f"Input shape was: {input_shape}"
            )

        self.input_shape = input_shape
        self.num_input_channels = input_shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=conv1_filters,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
            ),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                in_channels=conv1_filters,
                out_channels=conv2_filters,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
            ),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        mock_input = torch.rand((1, *input_shape))
        mock_conv_out = self.conv(mock_input)
        in_features = mock_conv_out.shape[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_features, n_features_linear),
            nn.ReLU(),
            nn.Linear(n_features_linear, n_features_linear),
            nn.ReLU(),
            nn.Linear(n_features_linear, n_components),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.encoder(x)
        return x
