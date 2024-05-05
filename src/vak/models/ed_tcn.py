"""
"""

from __future__ import annotations

import torch

from .. import metrics, nets
from .decorator import model
from .frame_classification_model import FrameClassificationModel


@model(family=FrameClassificationModel)
class ED_TCN:
    """Encoder-Decoder Temporal Convolutional Network model
    for the frame classification task.
    As described in [1]_. This adaptation adds convolutional
    blocks as a front end to extract features from the input.

    Attributes
    ----------
    network : vak.nets.ED_TCN
        Encoder-Decoder Temporal Convolutional Network architecture.
    loss: torch.nn.CrossEntropyLoss
        Standard cross-entropy loss
    optimizer: torch.optim.Adam
        Adam optimizer.
    metrics: dict
        Mapping string names to the following metrics:
        ``vak.metrics.Accuracy``, ``vak.metrics.Levenshtein``,
        ``vak.metrics.CharacterErrorRate``, ``torch.nn.CrossEntropyLoss``.

    References
    ----------
    .. [1] Lea, C., Flynn, M. D., Vidal, R., Reiter, A., & Hager, G. D. (2017).
       Temporal convolutional networks for action segmentation and detection.
       In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 156-165).
    """

    network = nets.ED_TCN
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {
        "acc": metrics.Accuracy,
        "levenshtein": metrics.Levenshtein,
        "character_error_rate": metrics.CharacterErrorRate,
        "loss": torch.nn.CrossEntropyLoss,
    }
    default_config = {"optimizer": {"lr": 0.003}}
