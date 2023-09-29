from __future__ import annotations

import warnings
import math
import torch
import numpy as np
# vak.nn.loss.vae
def vae_loss(
    x: torch.Tensor,
    z: torch.Tensor,
    x_rec: torch.Tensor,
    latent_dist: torch.Tensor,
    model_precision: float,
    z_dim: int
):
    pi = torch.tensor(math.pi)
    x_dim = x.shape
    elbo = -0.5 * ( torch.sum( torch.pow(z, 2) ) + z_dim * torch.log( 2 * pi ))
    # E_{q(z|x)} p(x|z)
    pxz_term = -0.5 * x_dim * (torch.log(2 * pi / model_precision))
    l2s = torch.sum( torch.pow( x.view( x.shape[0], -1 ) - x_rec, 2), dim=1)
    pxz_term = pxz_term - 0.5 * model_precision * torch.sum(l2s)
    elbo = elbo + pxz_term
    # H[q(z|x)]
    elbo = elbo + torch.sum(latent_dist.entropy())
    return elbo

class VaeLoss(torch.nn.Module):
    """"""

    def __init__(
        self,
        return_latent_rec: bool = False,
        model_precision: float = 10.0,
        z_dim: int = 32
    ):
        super().__init__()
        self.return_latent_rec = return_latent_rec
        self.model_precision = model_precision
        self.z_dim = z_dim

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        x_rec: torch.Tensor,
        latent_dist: torch.Tensor,
    ):
        x_shape = x.shape
        elbo = vae_loss(x=x, z=z, x_rec=x_rec, latent_dist=latent_dist, model_precision=self.model_precision, z_dim=self.z_dim)
        if self.return_latent_rec:
            return -elbo, z.detach().cpu().numpy(), \
                x_rec.view(-1, x_shape[0], x_shape[1]).detach().cpu().numpy()
        return -elbo
