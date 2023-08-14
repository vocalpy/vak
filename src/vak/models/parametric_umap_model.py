"""Parametric UMAP model, as described in [1]_.

Code adapted from implementation by @elyxlz
https://github.com/elyxlz/umap_pytorch
with changes made by Tim Sainburg:
https://github.com/lmcinnes/umap/issues/580#issuecomment-1368649550.
"""
from __future__ import annotations

import pathlib
from typing import Callable, ClassVar, Type

import pytorch_lightning as lightning
import torch
import torch.utils.data

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
        network: dict | None = None,
        loss: torch.nn.Module | Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict[str:Type] | None = None,
    ):
        super().__init__(
            network=network, loss=loss, optimizer=optimizer, metrics=metrics
        )
        self.encoder = network["encoder"]
        self.decoder = network.get("decoder", None)

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        (edges_to_exp, edges_from_exp) = batch
        embedding_to, embedding_from = self.encoder(
            edges_to_exp
        ), self.encoder(edges_from_exp)

        if self.decoder is not None:
            reconstruction = self.decoder(embedding_to)
            before_encoding = edges_to_exp
        else:
            reconstruction = None
            before_encoding = None
        loss_umap, loss_reconstruction, loss = self.loss(
            embedding_to, embedding_from, reconstruction, before_encoding
        )
        self.log("train_umap_loss", loss_umap)
        if loss_reconstruction:
            self.log("train_reconstruction_loss", loss_reconstruction)
        # note if there's no ``loss_reconstruction``, then ``loss`` == ``loss_umap``
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (edges_to_exp, edges_from_exp) = batch
        embedding_to, embedding_from = self.encoder(
            edges_to_exp
        ), self.encoder(edges_from_exp)

        if self.decoder is not None:
            reconstruction = self.decoder(embedding_to)
            before_encoding = edges_to_exp
        else:
            reconstruction = None
            before_encoding = None
        loss_umap, loss_reconstruction, loss = self.loss(
            embedding_to, embedding_from, reconstruction, before_encoding
        )
        self.log("val_umap_loss", loss_umap, on_step=True)
        if loss_reconstruction:
            self.log(
                "val_reconstruction_loss", loss_reconstruction, on_step=True
            )
        # note if there's no ``loss_reconstruction``, then ``loss`` == ``loss_umap``
        self.log("val_loss", loss, on_step=True)

    @classmethod
    def from_config(cls, config: dict):
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
        return cls(
            network=network, optimizer=optimizer, loss=loss, metrics=metrics
        )


class ParametricUMAPDatamodule(lightning.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )


class ParametricUMAP:
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module | None = None,
        n_neighbors: int = 10,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        num_epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers: int = 16,
        random_state: int | None = None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric

        self.lr = lr
        self.num_epochs = num_epochs

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state

        self.model = ParametricUMAPModel(self.encoder, min_dist=self.min_dist)

    def fit(
        self,
        trainer: lightning.Trainer,
        dataset_path: str | pathlib.Path,
        transform=None,
    ):
        from vak.datasets.parametric_umap import ParametricUMAPDataset

        dataset = ParametricUMAPDataset.from_dataset_path(
            dataset_path,
            "train",
            self.n_neighbors,
            self.metric,
            self.random_state,
            self.num_epochs,
            transform,
        )
        trainer.fit(
            model=self.model,
            datamodule=ParametricUMAPDatamodule(
                dataset, self.batch_size, self.num_workers
            ),
        )

    @torch.no_grad()
    def transform(self, X):
        embedding = self.model.encoder(X).detach().cpu().numpy()
        return embedding

    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()
