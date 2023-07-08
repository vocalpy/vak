"""Parametric UMAP model, as described in [1]_.

Code adapted from implementation by @elyxlz
https://github.com/elyxlz/umap_pytorch
with changes made by Tim Sainburg:
https://github.com/lmcinnes/umap/issues/580#issuecomment-1368649550.
"""
from __future__ import annotations

from typing import Callable, ClassVar, Type

import torch
from torch.nn.functional import mse_loss

from umap.umap_ import find_ab_params

from . import base
from .definition import ModelDefinition
from .registry import model_family


@model_family
class ParametricUMAPModel(base.Model):
    """Parametric UMAP model, as described in [1]_.

    Notes
    -----
    Code adapted from implementation by @elyxlz
    https://github.com/elyxlz/umap_pytorch
    with changes made by Tim Sainburg:
    https://github.com/lmcinnes/umap/issues/580#issuecomment-1368649550.

    References
    ----------
    .. [1] Sainburg, T., McInnes, L., & Gentner, T. Q. (2021).
       Parametric UMAP embeddings for representation and semisupervised learning.
       Neural Computation, 33(11), 2881-2907.
       https://direct.mit.edu/neco/article/33/11/2881/107068.
    """
    definition: ClassVar[ModelDefinition]

    def __init__(
        self,
        network: torch.nn.Module | dict[str: torch.nn.Module] | None = None,
        loss: torch.nn.Module | Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict[str: Type] | None = None,
        beta: float = 1.0,
        min_dist: float = 0.1,
        negative_sample_rate: int = 5,
    ):
        super().__init__(network=network, loss=loss,
                         optimizer=optimizer, metrics=metrics)
        self.encoder = network['encoder']
        self.decoder = network.get('decoder', None)
        self.beta = beta  # weight for reconstruction loss
        self._a, self._b = find_ab_params(1.0, min_dist)
        self.negative_sample_rate = negative_sample_rate

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        (edges_to_exp, edges_from_exp) = batch
        embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
        encoder_loss = self.loss(embedding_to, embedding_from, self._a, self._b,
                                 edges_to_exp.shape[0], negative_sample_rate=self.negative_sample_rate)
        self.log("train_umap_loss", encoder_loss)

        if self.decoder is not None:
            recon = self.decoder(embedding_to)
            recon_loss = mse_loss(recon, edges_to_exp)
            self.log("train_recon_loss", recon_loss)
            return encoder_loss + self.beta * recon_loss
        else:
            return encoder_loss

    def validation_step(self, batch, batch_idx):
        (edges_to_exp, edges_from_exp) = batch
        embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
        encoder_loss = self.loss(embedding_to, embedding_from, self._a, self._b,
                                 edges_to_exp.shape[0], negative_sample_rate=self.negative_sample_rate)
        self.log("val_umap_loss", encoder_loss, on_step=True)

        if self.decoder is not None:
            recon = self.decoder(embedding_to)
            recon_loss = mse_loss(recon, edges_to_exp)
            self.log("val_recon_loss", recon_loss, on_step=True)
            return encoder_loss + self.beta * recon_loss
        else:
            return encoder_loss

    @classmethod
    def from_config(cls,
                    config: dict,
                    beta: float = 1.0,
                    min_dist: float = 0.1,
                    negative_sample_rate: int = 5,
                    ):
        """Return an initialized model instance from a config ``dict``

        Parameters
        ----------
        config : dict
            Returned by calling :func:`vak.config.models.map_from_path`
            or :func:`vak.config.models.map_from_config_dict`.

        Returns
        -------
        cls : vak.models.base.Model
            An instance of the model with its attributes
            initialized using parameters from ``config``.
        """
        network, loss, optimizer, metrics = cls.attributes_from_config(config)
        return cls(network=network,
                   optimizer=optimizer,
                   loss=loss,
                   metrics=metrics,
                   beta=beta,
                   min_dist=min_dist,
                   negative_sample_rate=negative_sample_rate,
                   )
