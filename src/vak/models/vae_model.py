from __future__ import annotations

import pathlib
from typing import Callable, ClassVar, Type

import pytorch_lightning as lightning
import torch
import torch.utils.data
from torch import nn
from operator import itemgetter

from .registry import model_family
from . import base
from .definition import ModelDefinition

@model_family
class VAEModel(base.Model):
    definition: ClassVar[ModelDefinition]
    def __init__(
        self,
        network: dict | None = None,
        loss: torch.nn.Module | Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict[str:Type] | None = None,
    ):
        super().__init__(
            network=network, loss=loss, optimizer=optimizer, metrics=metrics
        )

    def forward(self, x):
        out, _ = self.network(x)
        return out

    def encode(self, x):
        return self.network.encoder(x)
    
    def decode(self, x):
        return self.network.decoder(x)

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch: tuple, batch_idx: int):
        """
        """
        x = batch[0]
        out, z, latent_dist= self.network(x)
        loss = self.loss(x, z, out, latent_dist)
        self.log("train_loss", loss)
        return loss
    
    def training_step(self, batch: tuple, batch_idx: int):
        """
        """
        x = batch[0]
        x = batch[0]
        out, _ = self.network(x)
        z, latent_dist  = itemgetter('z', 'latent_dist')(_)
        loss = self.loss(x, z, out, latent_dist)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        x = batch["frames"]
        x = batch[0]
        out, _ = self.network(x)
        z, latent_dist  = itemgetter('z', 'latent_dist')(_)
        for metric_name, metric_callable in self.metrics.items():
            if metric_name == "loss":
                self.log(
                    f"val_{metric_name}",
                    metric_callable(x, z, out, latent_dist),
                    batch_size=1,
                    on_step=True,
                )
            elif metric_name == "acc":
                self.log(
                    f"val_{metric_name}",
                    metric_callable(out, x),
                    batch_size=1,
                    on_step=True,
                )

    @classmethod
    def from_config(
        cls, config: dict
    ):
        network, loss, optimizer, metrics = cls.attributes_from_config(config)
        return cls(
            network=network,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )