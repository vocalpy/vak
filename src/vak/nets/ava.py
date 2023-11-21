from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal


class BottleneckLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]))

    def forward(self, x):
        return self.layer(x)


class AVA(nn.Module):
    """
    """
    def __init__(
        self,
        input_shape: tuple[int] = (1, 128, 128),
        encoder_channels: tuple[int] = (8, 8, 16, 16, 24, 24),
        fc_dims: tuple[int] = (1024, 256, 64),
        z_dim: int = 32,
    ):
        """
        """
        super().__init__()
        fc_dims = (*fc_dims, z_dim)
        encoder_channels = (*encoder_channels, z_dim)

        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.fc_view = (int(fc_dims[-1]),int(fc_dims[-1]/2),int(fc_dims[-1]/2))
        self.x_shape = input_shape[1:]
        self.x_dim = np.prod(self.x_shape)
        self.in_fc = int(self.x_dim / 2)

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

        # ---- build encoder bottleneck
        modules = []
        in_fc = self.in_fc
        for fc_dim in fc_dims[:-2]:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_fc, fc_dim),
                    nn.ReLU()
                )
            )
            in_fc = fc_dim
        self.encoder_bottleneck = nn.Sequential(*modules)

        self.mu_layer = BottleneckLayer(fc_dims[-3:])
        self.cov_factor_layer = BottleneckLayer(fc_dims[-3:])
        self.cov_diag_layer = BottleneckLayer(fc_dims[-3:])

        # ---- build decoder bottleneck
        modules = []
        fc_dims = fc_dims[::-1]
        for i in range(len(fc_dims)):
            out = self.in_fc if i == len(fc_dims) - 1 else fc_dims[i+1]
            modules.append(
                nn.Sequential(
                    nn.Linear(fc_dims[i], out),
                    nn.ReLU()
                )
            )
        self.decoder_bottleneck = nn.Sequential(*modules)

        # ---- build decoder
        modules = []
        decoder_channels = (*encoder_channels[-2::-1], self.in_channels)
        for i, out_channels in enumerate(decoder_channels):
            stride = 2 if out_channels == in_channels else 1
            output_padding = 1 if out_channels == in_channels else 0
            layers = [nn.BatchNorm2d(in_channels),
                      nn.ConvTranspose2d(
                          in_channels, out_channels=h_dim,
                          kernel_size=3, stride=stride, padding=1, output_padding=output_padding
                      )]
            if i != len(decoder_channels) - 1:
                layers.append(nn.ReLU())
            modules.append(nn.Sequential(*layers))
            in_channels = out_channels
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        """
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_bottleneck(x)
        mu = self.mu_layer(x)
        cov_factor = self.cov_factor_layer(x).unsqueeze(-1)
        cov_diag = torch.exp(self.cov_diag_layer(x))
        z, latent_dist = self.reparametrize(mu, cov_factor, cov_diag)
        return z, latent_dist

    def decode(self, z):
        """
        """
        z = self.decoder_bottleneck(z).view(-1, self.fc_view[0], self.fc_view[1], self.fc_view[2])
        z = self.decoder(z).view(-1, self.x_dim)
        return z

    @staticmethod
    def reparametrize(mu, cov_factor, cov_diag):
        latent_dist = LowRankMultivariateNormal(mu, cov_factor, cov_diag)
        z = latent_dist.rsample()
        return z, latent_dist

    def forward(self, x):
        z, latent_dist = self.encode(x)
        x_rec = self.decode(z).view(-1, self.x_shape[0], self.x_shape[1])
        return x_rec, z, latent_dist
