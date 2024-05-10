"""TweetyNet model [1]_.

.. [1] TweetyNet was described in:
   Cohen, Y., Nicholson, D. A., Sanchioni, A., Mallaber, E. K., Skidanova, V., & Gardner, T. J. (2022).
   Automated annotation of birdsong with a neural network that segments spectrograms. eLife 11: e63853.
   Paper: https://elifesciences.org/articles/63853
   Code: https://github.com/yardencsGitHub/tweetynet
"""

from __future__ import annotations

import torch

from .. import metrics, nets, nn
from .decorator import model
from .frame_classification_model import FrameClassificationModel


@model(family=FrameClassificationModel)
class TweetyNet:
    """TweetyNet model, as described in
    Cohen, Y., Nicholson, D. A., Sanchioni, A., Mallaber, E. K., Skidanova, V., & Gardner, T. J. (2022).
    Automated annotation of birdsong with a neural network that segments spectrograms. Elife, 11, e63853.
    https://elifesciences.org/articles/63853.

    Code adapted from
    https://github.com/yardencsGitHub/tweetynet.

    Attributes
    ----------
    network : vak.nets.TweetyNet
        Convolutional-bidirectional LSTM neural network architecture.
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
    TweetyNet was described in [1]_.

    ``TweetyNet`` is a type of windowed frame classification model,
    and this version built into ``vak`` relies on the
    ``FrameClassificationModel`` class.

    References
    ----------
    .. [1] Cohen, Y., Nicholson, D. A., Sanchioni, A., Mallaber, E. K., Skidanova, V., & Gardner, T. J. (2022).
       Automated annotation of birdsong with a neural network that segments spectrograms. eLife 11: e63853.
       Paper: https://elifesciences.org/articles/63853
       Code: https://github.com/yardencsGitHub/tweetynet
    """

    network = nets.TweetyNet
    loss = nn.loss.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {
        "acc": metrics.Accuracy,
        "levenshtein": metrics.Levenshtein,
        "character_error_rate": metrics.CharacterErrorRate,
        "loss": nn.loss.CrossEntropyLoss,
    }
    default_config = {"optimizer": {"lr": 0.003}}
