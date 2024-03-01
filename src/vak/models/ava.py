"""Autoencoded Vocal Analysis (AVA) model [1]_.
Code is adapted from [2]_.

.. [1] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
   Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.
   eLife, 10:e67855. https://doi.org/10.7554/eLife.67855

.. [2] https://github.com/pearsonlab/autoencoded-vocal-analysis
"""
from __future__ import annotations

import torch
from torchmetrics import KLDivergence
from .. import nets
from .decorator import model
from .vae_model import VAEModel
from ..nn.loss import VaeElboLoss


@model(family=VAEModel)
class AVA:
    """Autoencoded Vocal Analysis (AVA) model [1]_.

    .. [1] Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).
       Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.
       eLife, 10:e67855. https://doi.org/10.7554/eLife.67855
    """
    network = nets.AVA
    loss = VaeElboLoss
    optimizer = torch.optim.Adam
    metrics = {
        "loss": VaeElboLoss,
        "kl": KLDivergence
    }
    default_config = {"optimizer": {"lr": 1e-3}}
