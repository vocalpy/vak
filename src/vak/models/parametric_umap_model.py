"""Parametric UMAP model, as described in [1]_.

Code adapted from implementation by @elyxlz
https://github.com/elyxlz/umap_pytorch
with changes made by Tim Sainburg:
https://github.com/lmcinnes/umap/issues/580#issuecomment-1368649550.
"""

from __future__ import annotations

import pathlib
from typing import Callable, ClassVar, Type

import lightning
import torch
import torch.utils.data

from .definition import ModelDefinition
from .registry import model_family


@model_family
class ParametricUMAPModel(lightning.LightningModule):
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
        network: dict,
        loss: torch.nn.Module | Callable,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str:Type],
    ):
        super().__init__()
        self.network = torch.nn.ModuleDict(network)
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        (edges_to_exp, edges_from_exp) = batch
        embedding_to = self.network["encoder"](edges_to_exp)
        embedding_from = self.network["encoder"](edges_from_exp)

        if "decoder" in self.network:
            reconstruction = self.network["decoder"](embedding_to)
            before_encoding = edges_to_exp
        else:
            reconstruction = None
            before_encoding = None
        loss_umap, loss_reconstruction, loss = self.loss(
            embedding_to, embedding_from, reconstruction, before_encoding
        )
        self.log("train_umap_loss", loss_umap, on_step=True)
        if loss_reconstruction:
            self.log(
                "train_reconstruction_loss", loss_reconstruction, on_step=True
            )
        # note if there's no ``loss_reconstruction``, then ``loss`` == ``loss_umap``
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (edges_to_exp, edges_from_exp) = batch
        embedding_to = self.network["encoder"](edges_to_exp)
        embedding_from = self.network["encoder"](edges_from_exp)

        if "decoder" in self.network is not None:
            reconstruction = self.network["decoder"](embedding_to)
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

    def load_state_dict_from_path(self, ckpt_path):
        """Loads a model from the path to a saved checkpoint.

        Loads the checkpoint and then calls
        ``self.load_state_dict`` with the ``state_dict``
        in that chekcpoint.

        This method allows loading a state dict into an instance.
        It's necessary because `lightning.pytorch.LightningModule.load`` is a
        ``classmethod``, so calling that method will trigger
         ``LightningModule.__init__`` instead of running
        ``vak.models.Model.__init__``.

        Parameters
        ----------
        ckpt_path : str, pathlib.Path
            Path to a checkpoint saved by a model in ``vak``.
            This checkpoint has the same key-value pairs as
            any other checkpoint saved by a
            ``lightning.pytorch.LightningModule``.

        Returns
        -------
        None

        This method modifies the model state by loading the ``state_dict``;
        it does not return anything.
        """
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["state_dict"])


class ParametricUMAPDatamodule(lightning.pytorch.LightningDataModule):
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
        trainer: lightning.pytorch.Trainer,
        dataset_path: str | pathlib.Path,
        transform=None,
    ):
        from vak.datapipes.parametric_umap import ParametricUMAPDataset

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
