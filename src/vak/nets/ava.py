"""AVA variational autoencoder, as described in [1]_.
Code is adapted from [2]_.

.. [1] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
   Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.
   eLife, 10:e67855. https://doi.org/10.7554/eLife.67855

.. [2] https://github.com/pearsonlab/autoencoded-vocal-analysis
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal


class FullyConnectedLayers(nn.Module):
    """Module containing two fully-connected layers.

    This module is used to parametrize :math:`\mu`
    and :math:`\Sigma` in AVA.
    """
    def __init__(self, n_features: Sequence[int]):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_features[0], n_features[1]),
            nn.ReLU(),
            nn.Linear(n_features[1], n_features[2]))

    def forward(self, x):
        return self.layer(x)


class AVA(nn.Module):
    """AVA variational autoencoder, as described in [1]_.
    Code is adapted from [2]_.

    Attributes
    ----------
    input_shape
    in_channels
    x_shape
    x_dim
    encoder
    fc_view
    in_fc_dims
    shared_encoder_fc
    mu_fc
    cov_factor_fc
    cov_diag_fc
    decoder_fc
    decoder


    References
    ----------
    .. [1] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
       Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.
       eLife, 10:e67855. https://doi.org/10.7554/eLife.67855

    .. [2] https://github.com/pearsonlab/autoencoded-vocal-analysis
    """
    def __init__(
        self,
        input_shape: Sequence[int] = (1, 128, 128),
        encoder_channels: Sequence[int] = (8, 8, 16, 16, 24, 24, 32),
        fc_dims: Sequence[int] = (1024, 256, 64),
        z_dim: int = 32,
    ):
        """Initalize a new instance of
        an AVA variational autoencoder.

        Parameters
        ----------
        input_shape : Sequence
            Shape of input to network, a fixed size
            for all spectrograms.
            Tuple/list of integers, with dimensions
            (channels, frequency bins, time bins).
            Default is ``(1, 128, 128)``.
        encoder_channels : Sequence
            Number of channels in convolutional layers
            of encoder. Tuple/list of integers.
            Default is ``(8, 8, 16, 16, 24, 24, 32)``.
        fc_dims : Sequence
            Dimensionality of fully-connected layers.
            Tuple/list of integers.
            These values are used for the linear layers
            in the encoder (``self.shared_encoder_fc``)
            after passing through the convolutional layers,
            as well as the linear layers
            that are used to parametrize :math:`\mu` and
            :math:`\Sigma`.
            Default is (1024, 256, 64).
        z_dim : int
            Dimensionality of latent space.
            Default is 32.
        """
        super().__init__()

        self.input_shape = input_shape
        self.in_channels = int(input_shape[0])
        self.x_shape = input_shape[1:]  # channels * hide * width
        self.x_dim = int(np.prod(self.x_shape))

        # ---- build encoder
        modules = []
        in_channels = self.in_channels
        for out_channels in encoder_channels:
            # AVA uses stride=2 when out_channels == in_channels
            stride = 2 if out_channels == in_channels else 1
            modules.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=3, stride=stride, padding=1
                    ),
                    nn.ReLU()
                )
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*modules)

        # we compute shapes dynamically to make code more general
        # we could compute this using equations for conv shape etc. to avoid running tensor through encoder
        dummy_inp = torch.rand(1, *input_shape)
        out = self.encoder(dummy_inp)
        self.fc_view = tuple(out.shape[1:])
        out = torch.flatten(out, start_dim=1)
        self.in_fc_dims = out.shape[1]

        # ---- build shared fully-connected layers of encoder
        modules = []
        in_features = self.in_fc_dims
        for out_features in fc_dims[:-1]:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU()
                )
            )
            in_features = out_features
        self.shared_encoder_fc = nn.Sequential(*modules)

        fc_features = (*fc_dims[-2:], z_dim)
        self.mu_fc = FullyConnectedLayers(fc_features)
        self.cov_factor_fc = FullyConnectedLayers(fc_features)
        self.cov_diag_fc = FullyConnectedLayers(fc_features)

        # ---- build fully-connected layers of decoder
        modules = []
        decoder_dims = (*reversed(fc_dims), self.in_fc_dims)
        in_features = z_dim
        for i, out_features in enumerate(decoder_dims):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU()
                )
            )
            in_features = out_features
        self.decoder_fc = nn.Sequential(*modules)

        # ---- build decoder
        modules = []
        decoder_channels = (*reversed(encoder_channels[:-1]), self.in_channels)
        in_channels = encoder_channels[-1]
        for i, out_channels in enumerate(decoder_channels):
            stride = 2 if out_channels == in_channels else 1
            output_padding = 1 if out_channels == in_channels else 0
            layers = [nn.BatchNorm2d(in_channels),
                      nn.ConvTranspose2d(
                          in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=1, output_padding=output_padding
                      )]
            if i < len(decoder_channels) - 1:
                layers.append(nn.ReLU())
            modules.append(nn.Sequential(*layers))
            in_channels = out_channels
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        """Encode a spectrogram ``x``
        by mapping it to a vector :math:`z`
        in latent space.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        z : torch.Tensor
        latent_dist : torch.Tensor
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.shared_encoder_fc(x)
        mu = self.mu_fc(x)
        cov_factor = self.cov_factor_fc(x).unsqueeze(-1)  # Last dimension is rank \Sigma = 1
        cov_diag = torch.exp(self.cov_diag_fc(x))  # cov_diag must be positive
        z, latent_dist = self.reparametrize(mu, cov_factor, cov_diag)
        return z, latent_dist

    def decode(self, z):
        """Decode a latent space vector ``z``,
        mapping it back to a spectrogram :math:`x`
        in the space of spectrograms :math:`\mathcal{X}`.

        Parameters
        ----------
        z : torch.Tensor
            Output of encoder, with dimensions
            (batch size, latent space size).

        Returns
        -------
        x : torch.Tensor
            Output of decoder, with shape
            (batch, channel, frequency bins, time bins).
        """
        x = self.decoder_fc(z).view(-1, *self.fc_view)
        x = self.decoder(x).view(-1, *self.input_shape)
        return x

    @staticmethod
    def reparametrize(mu, cov_factor, cov_diag):
        """Sample a latent distribution
        to get the latent embedding :math:`z`.

        Method that encapsulates the reparametrization trick.

        Parameters
        ----------
        mu : torch.Tensor
        cov_factor : torch.Tensor
        cov_diag : torch.Tensor

        Returns
        -------
        z : torch.Tensor
        latent_dist : LowRankMultivariateNormal
        """
        latent_dist = LowRankMultivariateNormal(mu, cov_factor, cov_diag)
        z = latent_dist.rsample()
        return z, latent_dist

    def forward(self, x):
        """Pass a spectrogram ``x``
        through the variational autoencoder:
        encode, then decode.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        x_rec : torch.Tensor
            Reconstruction of ``x``,
            output of the decoder.
        z : torch.Tensor
            Latent space embedding of ``x``.
        latent_dist : LowRankMultivariateNormal
            Distribution parametrized
            by the output of the encoder.
        """
        z, latent_dist = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z, latent_dist
