from __future__ import annotations

import torch

from .. import metrics, nets
from .decorator import model
from .vae_model import VAEModel
from ..nn.loss import VaeElboLoss

@model(family=VAEModel)
class AVA:
    """
    """
    network = nets.Ava
    loss = VaeElboLoss
    optimizer = torch.optim.Adam
    metrics = {
        "loss": VaeElboLoss,
        "kl": torch.nn.functional.kl_div
    }
    default_config = {"optimizer": {"lr": 0.003}}