from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal


class FullyConnectedLayers(nn.Module):
    def __init__(self, n_features: Sequence[int]):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_features[0], n_features[1]),
            nn.ReLU(),
            nn.Linear(n_features[1], n_features[2]))

    def forward(self, x):
        return self.layer(x)


class AVA(nn.Module):
    """
    """
    def __init__(
        self,
        input_shape: Sequence[int] = (1, 128, 128),
        encoder_channels: Sequence[int] = (8, 8, 16, 16, 24, 24, 32),
        fc_dims: Sequence[int] = (1024, 256, 64),
        z_dim: int = 32,
    ):
        """
        """
        super().__init__()

        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.x_shape = input_shape[1:]  # channels * hide * width
        self.x_dim = np.prod(self.x_shape)

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
        """
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
        """
        """
        z = self.decoder_fc(z).view(-1, *self.fc_view)
        z = self.decoder(z).view(-1, *self.x_shape)
        return z

    @staticmethod
    def reparametrize(mu, cov_factor, cov_diag):
        latent_dist = LowRankMultivariateNormal(mu, cov_factor, cov_diag)
        z = latent_dist.rsample()
        return z, latent_dist

    def forward(self, x):
        z, latent_dist = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z, latent_dist
