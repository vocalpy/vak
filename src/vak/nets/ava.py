from __future__ import annotations

import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal

class Ava(nn.Module):
    """
    """
    def __init__(
        self,
        hidden_dims: list[int] = [8, 8, 16, 16, 24, 24, 32],
		fc_dims: list[int] = [1024, 256, 64, 32],
		in_channels: int = 1,
		in_fc: int = 8192,
		x_shape: tuple = (128, 128)	
    ):
        """
        """
        super().__init__()
        self.in_fc = in_fc
		self.in_channels = in_channels
		self.x_shape = x_shape 
		self.x_dim = torch.prod(x_shape)
		modules = []
		for h_dim in hidden_dims:
			stride = 2 if h_dim == in_channels else 1
            modules.append(
                nn.Sequential(
					nn.BatchNorm2d(in_channels),
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=stride, padding=1),
                    nn.ReLU())
            )
            in_channels = h_dim
		
		self.encoder = nn.Sequential(*modules)
		
		modules = []
		for fc_dim in fc_dims[:-2]:
            modules.append(
                nn.Sequential(
					nn.Linear(in_fc, fc_dim),
                    nn.ReLU())
            )
            in_fc = fc_dim
		self.encoder_bottleneck = nn.Sequential(*modules)

		self.mu_layer = nn.Sequential(
			nn.Linear(fc_dims[-3], fc_dims[-2]),
            nn.ReLU(),
			nn.Linear(fc_dims[-2], fc_dims[-1]))
		
		self.u_layer = nn.Sequential(
			nn.Linear(fc_dims[-3], fc_dims[-2]),
            nn.ReLU(),
			nn.Linear(fc_dims[-2], fc_dims[-1]))
		
		self.d_layer = nn.Sequential(
			nn.Linear(fc_dims[-3], fc_dims[-2]),
            nn.ReLU(),
			nn.Linear(fc_dims[-2], fc_dims[-1]))

		fc_dims.reverse()
		modules = []
		for i in range(len(fc_dims)):
			out = self.fc_in if i == len(fc_dims) else fc_dims[i+1]
            modules.append(
                nn.Sequential(
					nn.Linear(fc_dims[i], out),
                    nn.ReLU())
            )
		self.decoder_bottleneck = nn.Sequential(*modules)
        
		hidden_dims.reverse()
		modules = []
		for i, h_dim in enumerate(hidden_dims):
			stride = 2 if h_dim == in_channels else 1
			output_padding = 1 if h_dim == in_channels else 0
            modules.append(
                nn.Sequential(
					nn.BatchNorm2d(in_channels),
                    nn.ConvTranspose2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
                    nn.ReLU() if i != len(hidden_dims))
            )
            in_channels = h_dim

		self.decoder = nn.Sequential(*modules)

    def encode(self, x):
		"""
		"""
		x = self.encoder(x.unsqueeze(self.in_channels)).view(-1, self.in_fc)
		x = self.encoder_bottleneck(x)
		mu = self.mu_layer(x)
		u = self.u_layer(x).unsqueeze(-1)
		d = torch.exp(self.d_layer(x))
		z, latent_dist = self.reparametrize(mu, u, d)
		return z, latent_dist


    def decode(self, z):
		"""
		"""
		z = self.decoder_bottleneck(z).view(-1,32,16,16)
		z = self.decoder(z).view(-1, x_dim)
		return z

    def reparametrize(self, mu, u, d):
        latent_dist = LowRankMultivariateNormal(mu, u, d)
		z = latent_dist.rsample()
        return z, latent_dist


	def forward(self, x, return_latent_rec=False):
		z, latent_dist = self.encode(x)
		x_rec = self.decode(z)
		return x_rec, {'z': z, 'latent_dist': latent_dist,}