from __future__ import annotations

import math

import numpy as np
import torch


PI = torch.tensor(math.pi)

def vae_elbo_loss(
    x: torch.Tensor,
    z: torch.Tensor,
    x_rec: torch.Tensor,
    latent_dist: torch.Tensor,
    model_precision: float,
    z_dim: int
):
    # E_{q(z|x)} p(z)
    elbo = -0.5 * ( torch.sum( torch.pow(z, 2) ) + z_dim * torch.log( 2 * PI ))

    # E_{q(z|x)} p(x|z)
    x_dim = np.prod(x.shape[1:])
    pxz_term = -0.5 * x_dim * (torch.log(2 * PI / model_precision))
    l2s = torch.sum(torch.pow(x - x_rec, 2), dim=1)
    pxz_term = pxz_term - 0.5 * model_precision * torch.sum(l2s)
    elbo = elbo + pxz_term

    # H[q(z|x)]
    elbo = elbo + torch.sum(latent_dist.entropy())
    return -elbo


class VaeElboLoss(torch.nn.Module):
    """"""

    def __init__(
        self,
        model_precision: float = 10.0,
        z_dim: int = 32
    ):
        super().__init__()
        self.model_precision = model_precision
        self.z_dim = z_dim

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        x_rec: torch.Tensor,
        latent_dist: torch.Tensor,
    ):
        return vae_elbo_loss(
            x=x, z=z, x_rec=x_rec,
            latent_dist=latent_dist, model_precision=self.model_precision,
            z_dim=self.z_dim
        )

