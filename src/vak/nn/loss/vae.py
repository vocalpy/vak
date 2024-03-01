"""Evidence Lower Bound (ELBO) loss for a Variational Auto-Encpoder,
as used with the Autoencoded Vocal Analysis (AVA) model [1]_.
Code is adapted from [2]_.

.. [1] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
   Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.
   eLife, 10:e67855. https://doi.org/10.7554/eLife.67855

.. [2] https://github.com/pearsonlab/autoencoded-vocal-analysis
"""

from __future__ import annotations

import math

import numpy as np
import torch


PI = torch.tensor(math.pi)

def vae_elbo_loss(
    x: torch.Tensor,
    z: torch.Tensor,
    x_rec: torch.Tensor,
    latent_dist: torch.distributions.LowRankMultivariateNormal,
    model_precision: float,
    z_dim: int
) -> torch.Tensor:
    """Evidence Lower Bound (ELBO) loss for a Variational Auto-Encpoder,
    as used with the Autoencoded Vocal Analysis (AVA) model [1]_.

    Notes
    -----
    Code is adapted from [2]_.

    References
    ----------
    .. [1] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
       Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.
       eLife, 10:e67855. https://doi.org/10.7554/eLife.67855

    .. [2] https://github.com/pearsonlab/autoencoded-vocal-analysis

    Parameters
    ----------
    x : torch.Tensor
    z : torch.Tensor
    x_rec : torch.Tensor
    latent_dist
    model_precision : float
    z_dim : int
        Dimensionality of latent space

    Returns
    -------

    """
    # E_{q(z|x)} p(z)
    elbo = -0.5 * (torch.sum(torch.pow(z, 2) ) + z_dim * torch.log( 2 * PI ))

    # E_{q(z|x)} p(x|z)
    x_dim = np.prod(x.shape[1:])
    pxz_term = -0.5 * x_dim * (torch.log(2 * PI / model_precision))
    l2s = torch.sum(
        torch.pow(
            x.view(x.shape[0], -1) - x_rec.view(x_rec.shape[0], -1),
            2),
        dim=1
    )
    pxz_term = pxz_term - 0.5 * model_precision * torch.sum(l2s)
    elbo = elbo + pxz_term

    # H[q(z|x)]
    elbo = elbo + torch.sum(latent_dist.entropy())
    return -elbo


class VaeElboLoss(torch.nn.Module):
    """Evidence Lower Bound (ELBO) loss for a Variational Auto-Encpoder,
    as used with the Autoencoded Vocal Analysis (AVA) model [1]_.

    ELBO can be written as
    :math:`L(\phi, \theta; x) = \text{ln} p_{\theta}(x) - D_{KL}(q_{\phi}(z|x) || p_{\theta}(z|x))`
    where the first term is the *evidence* for :math:`x`
    and the second is the Kullback-Leibler divergence between
    :math:`q_{\phi}` and :math:`p_{\theta}`.

    Notes
    -----
    Code is adapted from [2]_.

    References
    ----------
    .. [1] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
       Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.
       eLife, 10:e67855. https://doi.org/10.7554/eLife.67855
    .. [2] https://github.com/pearsonlab/autoencoded-vocal-analysis
    """
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
        latent_dist: torch.distributions.LowRankMultivariateNormal,
    ):
        """Compute ELBO loss

        Parameters
        ----------
        x
        z
        x_rec
        latent_dist

        Returns
        -------

        """
        return vae_elbo_loss(
            x=x, z=z, x_rec=x_rec,
            latent_dist=latent_dist, model_precision=self.model_precision,
            z_dim=self.z_dim
        )

