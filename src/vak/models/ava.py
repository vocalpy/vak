from __future__ import annotations

import torch

from .. import metrics, nets
from .decorator import model
from .vae_model import VAEModel
from ..nn.loss import VaeLoss

@model(family=VAEModel)
class AVA:
    """
    """
    network = Ava
    loss = VaeLoss
    optimizer = torch.optim.Adam
    metrics = {
        "loss": VaeLoss,
        "kl": torch.nn.functional.kl_div
    }
    default_config = {"optimizer": {"lr": 0.003}}