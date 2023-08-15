"""A LightningModule that represents a task
where a model predicts a label for each frame
in a time series, e.g., each time bin in
a window from a spectrogram."""
from __future__ import annotations

import logging
from typing import Callable, ClassVar, Mapping

import torch

from .. import transforms
from ..common import labels
from . import base
from .definition import ModelDefinition
from .registry import model_family

logger = logging.getLogger(__name__)


@model_family
class FrameClassificationModel(base.Model):
    """Class that represents a family of neural network models
    that predicts a label for each frame in a time series,
    e.g., each time bin in a window from a spectrogram.

    The task of predicting a label for each frame in a series
    is one way of predicting annotations for a vocalization,
    where the annotations consist of a sequence
    of segments, each with an onset, offset, and label.
    The model maps the spectrogram window
    to a vector of labels for each frame, i.e., each time bin.

    To annotate a vocalization with such a model,
    the spectrogram is converted into a batch of
    consecutive non-overlapping windows,
    for which the model produces predictions.
    These predictions are then concatenated
    into a vector of labeled frames,
    from which the segments can be recovered.

    Post-processing can be applied to the vector
    to clean up noisy predictions
    before recovering the segments.

    Attributes
    ----------
    network : torch.nn.Module, dict
        An instance of a ``torch.nn.Module``
        that implements a neural network,
        or a ``dict`` that maps human-readable string names
        to a set of such instances.
    loss : torch.nn.Module, callable
        An instance of a ``torch.nn.Module``
        that implements a loss function,
        or a callable Python function that
        computes a scalar loss.
    optimizer : torch.optim.Optimizer
        An instance of a ``torch.optim.Optimizer`` class
        used with ``loss`` to optimize
        the parameters of ``network``.
    metrics : dict
        A ``dict`` that maps human-readable string names
        to ``Callable`` functions, used to measure
        performance of the model.
    post_tfm : callable
        Post-processing transform applied to predictions.
    labelmap : dict-like
        That maps human-readable labels to integers predicted by network.
    eval_labelmap : dict-like
        Mapping from labels to integers predicted by network
        that is used by ``validation_step``.
        This is used when mapping from network outputs back to labels
        to compute metrics that require strings, such as edit distance.
        If ``labelmap`` contains keys with multiple characters,
        this will be ``labelmap`` re-mapped so that all labels have
        single characters (except "unlabeled"), to avoid artificially
        changing the edit distance.
        See https://github.com/vocalpy/vak/issues/373 for more detail.
        If all keys (except "unlabeled") are single-character,
        then ``eval_labelmap`` will just be ``labelmap``.
    to_labels_eval : vak.transforms.frame_labels.ToLabels
        Instance of :class:`~vak.transforms.frame_labels.ToLabels`
        that uses ``eval_labelmap`` to convert labeled timebins
        to string labels inside of ``validation_step``,
        for computing edit distance.
    """

    definition: ClassVar[ModelDefinition]

    def __init__(
        self,
        labelmap: Mapping,
        network: torch.nn.Module | dict | None = None,
        loss: torch.nn.Module | Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict | None = None,
        post_tfm: Callable | None = None,
    ):
        """Initialize a new instance of a
        :class:`~vak.models.frame_classification_model.FrameClassificationModel`.

        Parameters
        ----------
        labelmap : dict-like
            That maps human-readable labels to integers predicted by network.
        network : torch.nn.Module, dict
            An instance of a :class:`torch.nn.Module`
            that implements a neural network,
            or a ``dict`` that maps human-readable string names
            to a set of such instances.
        loss : torch.nn.Module, callable
            An instance of a :class:`torch.nn.Module`
            that implements a loss function,
            or a callable Python function that
            computes a scalar loss.
        optimizer : torch.optim.Optimizer
            An instance of a ``torch.optim.Optimizer`` class
            used with ``loss`` to optimize
            the parameters of ``network``.
        metrics : dict
            A ``dict`` that maps human-readable string names
            to ``Callable`` functions, used to measure
            performance of the model.
        post_tfm : callable
            Post-processing transform applied to predictions.
        """
        super().__init__(
            network=network, loss=loss, optimizer=optimizer, metrics=metrics
        )

        self.labelmap = labelmap
        # replace any multiple character labels in mapping
        # with single-character labels
        # so that we do not affect edit distance computation
        # see https://github.com/NickleDave/vak/issues/373
        labelmap_keys = [lbl for lbl in labelmap.keys() if lbl != "unlabeled"]
        if any(
            [len(label) > 1 for label in labelmap_keys]
        ):  # only re-map if necessary
            # (to minimize chance of knock-on bugs)
            logger.info(
                "Detected that labelmap has keys with multiple characters:"
                f"\n{labelmap_keys}\n"
                "Re-mapping labelmap used with to_labels_eval transform, using "
                "function vak.labels.multi_char_labels_to_single_char"
            )
            self.eval_labelmap = labels.multi_char_labels_to_single_char(
                labelmap
            )
        else:
            self.eval_labelmap = labelmap

        self.to_labels_eval = transforms.frame_labels.ToLabels(
            self.eval_labelmap
        )
        self.post_tfm = post_tfm

    def configure_optimizers(self):
        """Returns the model's optimizer.

        Method required by ``lightning.LightningModule``.
        This method returns the ``optimizer`` instance passed into ``__init__``.
        If None was passed in, an instance that was created
        with default arguments will be returned.
        """
        return self.optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through this model's network.

        Parameters
        ----------
        x : torch.Tensor
            Input to network, with shape that matches ``self.input_shape``.

        Returns
        -------
        out : torch.Tensor
            Output from network.
        """
        return self.network(x)

    def training_step(self, batch: tuple, batch_idx: int):
        """Perform one training step.

        Method required by ``lightning.LightningModule``.

        Parameters
        ----------
        batch : tuple
            A batch from a dataloader.
        batch_idx : int
            The index of this batch in the dataloader.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value computed by
            the loss function, ``self.loss``.
        """
        x, y = batch[0], batch[1]
        out = self.network(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        """Perform one validation step.

        Method required by ``lightning.LightningModule``.
        Logs metrics using ``self.log``

        Parameters
        ----------
        batch : tuple
            A batch from a dataloader.
        batch_idx : int
            The index of this batch in the dataloader.

        Returns
        -------
        None
        """
        x, y = batch["frames"], batch["frame_labels"]
        # remove "batch" dimension added by collate_fn to x
        # we keep for y because loss still expects the first dimension to be batch
        # TODO: fix this weirdness. Diff't collate_fn?
        if x.ndim in (5, 4):
            if x.shape[0] == 1:
                x = torch.squeeze(x, dim=0)
        else:
            raise ValueError(f"invalid shape for x: {x.shape}")

        out = self.network(x)
        # permute and flatten out
        # so that it has shape (1, number classes, number of time bins)
        # ** NOTICE ** just calling out.reshape(1, out.shape(1), -1) does not work, it will change the data
        out = out.permute(1, 0, 2)
        out = torch.flatten(out, start_dim=1)
        out = torch.unsqueeze(out, dim=0)
        # reduce to predictions, assuming class dimension is 1
        y_pred = torch.argmax(
            out, dim=1
        )  # y_pred has dims (batch size 1, predicted label per time bin)

        if "padding_mask" in batch:
            padding_mask = batch[
                "padding_mask"
            ]  # boolean: 1 where valid, 0 where padding
            # remove "batch" dimension added by collate_fn
            # because this extra dimension just makes it confusing to use the mask as indices
            if padding_mask.ndim == 2:
                if padding_mask.shape[0] == 1:
                    padding_mask = torch.squeeze(padding_mask, dim=0)
            else:
                raise ValueError(
                    f"invalid shape for padding mask: {padding_mask.shape}"
                )

            out = out[:, :, padding_mask]
            y_pred = y_pred[:, padding_mask]

        y_labels = self.to_labels_eval(y.cpu().numpy())
        y_pred_labels = self.to_labels_eval(y_pred.cpu().numpy())

        if self.post_tfm:
            y_pred_tfm = self.post_tfm(
                y_pred.cpu().numpy(),
            )
            y_pred_tfm_labels = self.to_labels_eval(y_pred_tfm)
            # convert back to tensor so we can compute accuracy
            y_pred_tfm = torch.from_numpy(y_pred_tfm).to(self.device)

        # TODO: figure out smarter way to do this
        for metric_name, metric_callable in self.metrics.items():
            if metric_name == "loss":
                self.log(
                    f"val_{metric_name}",
                    metric_callable(out, y),
                    batch_size=1,
                    on_step=True,
                )
            elif metric_name == "acc":
                self.log(
                    f"val_{metric_name}",
                    metric_callable(y_pred, y),
                    batch_size=1,
                )
                if self.post_tfm:
                    self.log(
                        f"val_{metric_name}_tfm",
                        metric_callable(y_pred_tfm, y),
                        batch_size=1,
                        on_step=True,
                    )
            elif (
                metric_name == "levenshtein"
                or metric_name == "segment_error_rate"
            ):
                self.log(
                    f"val_{metric_name}",
                    metric_callable(y_pred_labels, y_labels),
                    batch_size=1,
                )
                if self.post_tfm:
                    self.log(
                        f"val_{metric_name}_tfm",
                        metric_callable(y_pred_tfm_labels, y_labels),
                        batch_size=1,
                        on_step=True,
                    )

    def predict_step(self, batch: tuple, batch_idx: int):
        """Perform one prediction step.

        Method required by ``lightning.LightningModule``.

        Parameters
        ----------
        batch : tuple
            A batch from a dataloader.
        batch_idx : int
            The index of this batch in the dataloader.

        Returns
        -------
        y_pred : dict
            Where the key is "source_path" and the value
            is the output of the network;
            "source_path" is the path to the file
            containing the spectrogram
            for which a prediction was generated.
        """
        x, source_path = batch["frames"].to(self.device), batch["source_path"]
        if isinstance(source_path, list) and len(source_path) == 1:
            source_path = source_path[0]
        # TODO: fix this weirdness. Diff't collate_fn?
        if x.ndim in (5, 4):
            if x.shape[0] == 1:
                x = torch.squeeze(x, dim=0)
        else:
            raise ValueError(f"invalid shape for x: {x.shape}")
        y_pred = self.network(x)
        return {source_path: y_pred}

    @classmethod
    def from_config(
        cls, config: dict, labelmap: Mapping, post_tfm: Callable | None = None
    ):
        """Return an initialized model instance from a config ``dict``

        Parameters
        ----------
        config : dict
            Returned by calling :func:`vak.config.models.map_from_path`
            or :func:`vak.config.models.map_from_config_dict`.
        post_tfm : callable
            Post-processing transformation.
            A callable applied to the network output.
            Default is None.

        Returns
        -------
        cls : vak.models.base.Model
            An instance of the model with its attributes
            initialized using parameters from ``config``.
        """
        network, loss, optimizer, metrics = cls.attributes_from_config(config)
        return cls(
            labelmap=labelmap,
            network=network,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            post_tfm=post_tfm,
        )
