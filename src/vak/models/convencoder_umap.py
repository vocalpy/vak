"""Parametric UMAP model, as described in [1]_, with a convolutional network as the encoder.

Code adapted from implementation by @elyxlz
https://github.com/elyxlz/umap_pytorch
with changes made by Tim Sainburg:
https://github.com/lmcinnes/umap/issues/580#issuecomment-1368649550.
"""
from __future__ import annotations

import math

import torch

from .. import (
    metrics,
    nets,
    nn,
)
from .parametric_umap_model import ParametricUMAPModel
from .decorator import model


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
        ``vak.metrics.SegmentErrorRate``, ``torch.nn.CrossEntropyLoss``.

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
    network = {'encoder': nets.ConvEncoder}
    loss = nn.UmapLoss
    optimizer = torch.optim.AdamW
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        'optimizer':
            {'lr': 1e-3},
    }


def next_power_of_2(x: int | float) -> int:
    """Compute the nearest power of 2 to a number.

    Used e.g. to pad an input
    to a convolutional neural network.

    Parameters
    ----------
    x : int, float
        A number :math:`x` for which we would like
        to find the nearest power of 2.

    Returns
    -------
    pow2 : int
        The nearest power of 2 to :math:`x`.
    """
    return 1 if x == 0 else 2**math.ceil(math.log2(x))


def get_default_padding(shape):
    """Get default padding for input to ConvEncoderUMAP model.

    Pads the input shape so that each dimension is a power of 2.
    """
    shape_pow2 = tuple(next_power_of_2(x) for x in shape)
    padding = (xpow2 - x for (xpow2, x) in zip(shape_pow2, shape))
    return padding
