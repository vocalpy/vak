"""Parametric UMAP model, as described in [1]_, with a convolutional network as the encoder.

Code adapted from implementation by @elyxlz
https://github.com/elyxlz/umap_pytorch
with changes made by Tim Sainburg:
https://github.com/lmcinnes/umap/issues/580#issuecomment-1368649550.
"""

from __future__ import annotations

import torch

from .. import metrics, nets, nn
from .decorator import model
from .parametric_umap_model import ParametricUMAPModel


@model(family=ParametricUMAPModel)
class ConvEncoderUMAP:
    """Parametric UMAP model, as described in [1]_,
    with a convolutional network as the encoder.

    Attributes
    ----------
    network : dict
        A dict with two keys, 'encoder' and 'decoder'.
        The 'encoder is vak.nets.ConvEncoder,
        an encoder with convolutional layers.
        The 'decoder' defaults to None.
    loss: torch.nn.CrossEntropyLoss
        Standard cross-entropy loss
    optimizer: torch.optim.Adam
        Adam optimizer.
    metrics: dict
        Mapping string names to the following metrics:
        ``vak.metrics.Accuracy``, ``vak.metrics.Levenshtein``,
        ``vak.metrics.CharacterErrorRate``, ``torch.nn.CrossEntropyLoss``.

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

    network = {"encoder": nets.ConvEncoder}
    loss = nn.UmapLoss
    optimizer = torch.optim.AdamW
    metrics = {
        "acc": metrics.Accuracy,
        "levenshtein": metrics.Levenshtein,
        "character_error_rate": metrics.CharacterErrorRate,
        "loss": torch.nn.CrossEntropyLoss,
    }
    default_config = {
        "optimizer": {"lr": 1e-3},
    }
