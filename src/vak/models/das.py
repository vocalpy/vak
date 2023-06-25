"""Deep Audio Segmenter (DAS) model [1]_.

.. [1] Deep Audio Segmenter (DAS) model was described in:
   Elsa Steinfath, Adrian Palacios, Julian Rottschäfer, Deniz Yuezak, Jan Clemens (2021).
   Fast and accurate annotation of acoustic signals with deep neural networks. eLife 10:e68837.
   Paper: https://doi.org/10.7554/eLife.68837
   Code: https://github.com/janclemenslab/das
"""
from __future__ import annotations


import torch

from .. import (
    metrics,
    nets
)
from .windowed_frame_classification_model import WindowedFrameClassificationModel
from .decorator import model


@model(family=WindowedFrameClassificationModel)
class DAS:
    """Deep Audio Segmenter (DAS) model [1]_.

    Attributes
    ----------
    network : vak.nets.DASNet
        Temporal convolutional neural network architecture.
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
    Deep Audio Segmenter (DAS) model was described in [1]_.

    ``DASNet`` is a type of windowed frame classification model,
    and this version built into ``vak`` relies on the
    ``WindowedFrameClassificationModel`` class.

    References
    ----------
    .. [1] Elsa Steinfath, Adrian Palacios, Julian Rottschäfer, Deniz Yuezak, Jan Clemens (2021).
       Fast and accurate annotation of acoustic signals with deep neural networks. eLife 10:e68837.
       Paper: https://doi.org/10.7554/eLife.68837
       Code: https://github.com/janclemenslab/das

    Examples
    --------

    The simplest way to make an instance of a model directly is to first make an instance
    of the neural network architecture, that requires the number of samples in the
    input audio window and the number of classes.
    Then make an instance of the model with the network instance and a mapping from
    integer outputs to string labels.

    >>> import vak
    >>> labelmap = {0: 'unlabeled', 1: 'sine', 2: 'pulse'}
    >>> net = vak.nets.DASNet(num_samples=8092, num_classes=len(labelmap))
    >>> model = vak.models.DAS(network=net, labelmap=labelmap)

    An instance of the network could also be made with the builder functions
    built into the DASNet module.
    >>> import vak
    >>> net = vak.nets.das.dasnet_bengalese_finch()
    >>> # make a mock labelmap to keep snippet brief
    >>> num_classes = vak.nets.das.net.DASNetBengaleseFinchConfig.num_classes
    >>> labelmap = {lbl:str(lbl) for lbl in range(num_classes)}
    >>> model = vak.models.DAS(network=net, labelmap=labelmap)
    """
    network = nets.DASNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        'optimizer':
# Saved Bengalese finch model has learning rate of 0.001. Comments in code suggest this is default.
# although model building function has default of 0.0005 AFAICT?
# https://github.com/janclemenslab/das/blob/3f3bf76b705e0960d5fd84f26033a4fa3cde2472/src/das/models.py#L46
# Don't find mention of it in paper methods
            {'lr': 0.001}
    }
