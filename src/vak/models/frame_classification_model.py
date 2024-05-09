"""A LightningModule that represents a task
where a model predicts a label for each frame
in a time series, e.g., each time bin in
a window from a spectrogram."""

from __future__ import annotations

import logging
from typing import Callable, Mapping

import lightning
import torch

from .. import transforms
from ..common import labels
from .registry import model_family

logger = logging.getLogger(__name__)


@model_family
class FrameClassificationModel(lightning.LightningModule):
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
        super().__init__()

        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

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

        Method required by ``lightning.pytorch.LightningModule``.
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

        Method required by ``lightning.pytorch.LightningModule``.

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
        frames = batch["frames"]
        frame_logits = self.network(frames)

        if (  # multi-class frame classificaton
            "multi_frame_labels" in batch and
            "binary_frame_labels" not in batch and
            "boundary_frame_labels" not in batch
            ):
                loss = self.loss(frame_logits, batch["multi_frame_labels"])
        elif (  # binary frame classification
            "binary_frame_labels" in batch and
            "multi_frame_labels" not in batch and
            "boundary_frame_labels" not in batch
            ):

                loss = self.loss(frame_logits, batch["binary_frame_labels"])
        elif (  # boundary "detection" -- i.e. different kind of binary frame classification
            "boundary_frame_labels" in batch and
            "multi_frame_labels" not in batch and
            "binary_frame_labels" not in batch
            ):
                loss = self.loss(frame_logits, batch["boundary_frame_labels"])
        elif (  # multi-class frame classification *and* boundary detection
            "multi_frame_labels" in batch and
            "boundary_frame_labels" in batch and
            "binary_frame_labels" not in batch
            ):
                multi_logits, boundary_logits = self.network(frames)
                multi_loss = self.loss(multi_logits, batch["multi_frame_labels"])
                boundary_loss = self.loss(boundary_logits, batch["boundary_frame_labels"])
                loss = multi_loss + boundary_loss

        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        """Perform one validation step.

        Method required by ``lightning.pytorch.LightningModule``.
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
        frames = batch["frames"]
        # remove "batch" dimension added by collate_fn to frames
        # TODO: fix this weirdness. Diff't collate_fn?
        if frames.ndim in (5, 4):
            if frames.shape[0] == 1:
                frames = torch.squeeze(frames, dim=0)
        else:
            raise ValueError(f"invalid shape for frames: {frames.shape}")

        if (  # multi-class frame classificaton
            "multi_frame_labels" in batch and
            "binary_frame_labels" not in batch and
            "boundary_frame_labels" not in batch
            ):
                target_types = ("multi_frame_labels",)
        elif (  # binary frame classification
            "binary_frame_labels" in batch and
            "multi_frame_labels" not in batch and
            "boundary_frame_labels" not in batch
            ):
                target_types = ("binary_frame_labels",)
        elif (  # boundary "detection" -- i.e. different kind of binary frame classification
            "boundary_frame_labels" in batch and
            "multi_frame_labels" not in batch and
            "binary_frame_labels" not in batch
            ):
                target_types = ("boundary_frame_labels",)
        elif (  # multi-class frame classification *and* boundary detection
            "multi_frame_labels" in batch and
            "boundary_frame_labels" in batch and
            "binary_frame_labels" not in batch
            ):
                target_types = ("multi_frame_labels", "boundary_frame_labels")
        if len(target_types) == 1:
            target = batch[target_types[0]]
        else:
            target = tuple(batch[target_types_] for target_types_ in target_types)

        if len(target_types) == 1:
            frame_logits = self.network(frames)
        else:
            frame_logits, boundary_logits = self.network(frames)

        # permute and flatten out
        # so that it has shape (1, number classes, number of time bins)
        # ** NOTICE ** just calling out.reshape(1, out.shape(1), -1) does not work, it will change the data
        frame_logits = frame_logits.permute(1, 0, 2)
        frame_logits = torch.flatten(frame_logits, start_dim=1)
        frame_logits = torch.unsqueeze(frame_logits, dim=0)
        # reduce to predictions, assuming class dimension is 1
        frame_preds = torch.argmax(
            frame_logits, dim=1
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

            frame_logits = frame_logits[:, :, padding_mask]
            frame_preds = frame_preds[:, padding_mask]

        if "multi_frame_labels" in target_types:
            multi_frame_labels_str = self.to_labels_eval(batch["multi_frame_labels"].cpu().numpy())
            frame_preds_str = self.to_labels_eval(frame_preds.cpu().numpy())

            if self.post_tfm:
                frame_preds_tfm = self.post_tfm(
                    frame_preds.cpu().numpy(),
                )
                frame_preds_tfm_str = self.to_labels_eval(frame_preds_tfm)
                # convert back to tensor so we can compute accuracy
                frame_preds_tfm = torch.from_numpy(frame_preds_tfm).to(self.device)

        for metric_name, metric_callable in self.metrics.items():
            if metric_name == "loss":
                if len(target_types) == 1:
                    self.log(
                        f"val_{metric_name}",
                        metric_callable(frame_logits, target),
                        batch_size=1,
                        on_step=True,
                        sync_dist=True,
                    )
                else:
                    multi_logits, boundary_logits = self.network(frames)
                    multi_loss = self.loss(multi_logits, batch["multi_frame_labels"])
                    boundary_loss = self.loss(boundary_logits, batch["boundary_frame_labels"])
                    loss = multi_loss + boundary_loss
                    self.log(
                        f"val_{metric_name}",
                        loss,
                        batch_size=1,
                        on_step=True,
                        sync_dist=True,
                    )
            elif metric_name == "acc":
                if len(target_types) == 1:
                    self.log(
                        f"val_{metric_name}",
                        metric_callable(frame_preds, target),
                        batch_size=1,
                        on_step=True,
                        sync_dist=True,
                    )
                    if self.post_tfm and "multi_frame_labels" in target_types:
                        self.log(
                            f"val_{metric_name}_tfm",
                            metric_callable(frame_preds_tfm, target),
                            batch_size=1,
                            on_step=True,
                            sync_dist=True,
                        )
                else:
                    # FIXME
                    pass
            elif (
                metric_name == "levenshtein"
                or metric_name == "character_error_rate"
            ) and "multi_frame_labels" in target_types:
                self.log(
                    f"val_{metric_name}",
                    # next line: convert to float to squelch warning from lightning
                    float(metric_callable(frame_preds_str, multi_frame_labels_str)),
                    batch_size=1,
                    on_step=True,
                    sync_dist=True,
                )
                if self.post_tfm:
                    self.log(
                        f"val_{metric_name}_tfm",
                        # next line: convert to float to squelch warning from lightning
                        float(metric_callable(frame_preds_tfm_str, multi_frame_labels_str)),
                        batch_size=1,
                        on_step=True,
                        sync_dist=True,
                    )

    def predict_step(self, batch: tuple, batch_idx: int):
        """Perform one prediction step.

        Method required by ``lightning.pytorch.LightningModule``.

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
        x, frames_path = batch["frames"].to(self.device), batch["frames_path"]
        if isinstance(frames_path, list) and len(frames_path) == 1:
            frames_path = frames_path[0]
        # TODO: fix this weirdness. Diff't collate_fn?
        if x.ndim in (5, 4):
            if x.shape[0] == 1:
                x = torch.squeeze(x, dim=0)
        else:
            raise ValueError(f"invalid shape for x: {x.shape}")
        y_pred = self.network(x)
        return {frames_path: y_pred}

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
