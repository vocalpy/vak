"""lightweight version of ``vak.models.TweetyNet`` used by ``vak`` unit tests
"""
import torch

from .. import metrics, nets
from .decorator import model
from .frame_classification_model import FrameClassificationModel


@model(family=FrameClassificationModel)
class TeenyTweetyNet:
    """lightweight version of ``vak.models.TweetyNet`` used by ``vak`` unit tests"""

    network = nets.TeenyTweetyNet
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {
        "acc": metrics.Accuracy,
        "levenshtein": metrics.Levenshtein,
        "segment_error_rate": metrics.SegmentErrorRate,
        "loss": torch.nn.CrossEntropyLoss,
    }
    default_config = {"optimizer": {"lr": 0.003}}
