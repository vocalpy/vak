from __future__ import annotations

import torch
from torchmetrics import KLDivergence
from .. import metrics, nets
from .decorator import model
from .vae_model import VAEModel
from ..nn.loss import VaeElboLoss

@model(family=VAEModel)
class AVA:
    """
    """
    network = nets.AVA
    loss = VaeElboLoss
    optimizer = torch.optim.Adam
    metrics = {
        "loss": VaeElboLoss,
        "kl": KLDivergence
    }
    default_config = {"optimizer": {"lr": 0.003}}