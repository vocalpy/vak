"""Modules that act as activation functions."""

import torch


class NormReLU(torch.nn.Module):
    """This module implements normalized ReLU activation,
    as used with the Encoder-Decoder Temporal Convolutional Model in [1]_.

    The output of the activation is normalized by the channel-wise maximum.

    .. [1] Lea, C., Flynn, M. D., Vidal, R., Reiter, A., & Hager, G. D. (2017).
       Temporal convolutional networks for action segmentation and detection.
       In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 156-165).
    """

    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(x)
        max_values = torch.max(torch.abs(x), dim=-1, keepdims=True)[0] + 1e-5
        x = x / max_values
        return x
