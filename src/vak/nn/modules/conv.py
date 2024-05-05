"""Modules that perform neural network convolutions."""

import torch
from torch.nn import functional as F


# NOTE: added 2023-03-04
# in this class, we detect when one extra pixel should be added on the bottom or right
# and specifically pad those, see line 75, ``if rows_odd or cols_odd:``.
class Conv2dTF(torch.nn.Conv2d):
    """Conv2d with padding behavior from Tensorflow

    Adapted from
    https://github.com/mlperf/inference/blob/16a5661eea8f0545e04c86029362e22113c2ec09/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py#L40
    as referenced in this issue:
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-507025011

    Used to maintain behavior of original implementation of TweetyNet that used Tensorflow 1.0 low-level API.

    Note there are issues with SAME convolution as performed by Tensorflow.
    See https://gist.github.com/Yangqing/47772de7eb3d5dbbff50ffb0d7a98964.
    """

    PADDING_METHODS = ("VALID", "SAME")

    def __init__(self, *args, **kwargs):
        # remove 'padding' from ``kwargs`` to avoid bug in ``torch`` => 1.7.2
        # see https://github.com/yardencsGitHub/tweetynet/issues/166
        kwargs_super = {k: v for k, v in kwargs.items() if k != "padding"}
        super(Conv2dTF, self).__init__(*args, **kwargs_super)
        padding = kwargs.get("padding", "SAME")
        if not isinstance(padding, str):
            raise TypeError(
                f"value for 'padding' argument should be a string, one of: {self.PADDING_METHODS}"
            )
        padding = padding.upper()
        if padding not in self.PADDING_METHODS:
            raise ValueError(
                f"value for 'padding' argument must be one of '{self.PADDING_METHODS}' but was: {padding}"
            )
        self.padding = padding

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0,
            (out_size - 1) * self.stride[dim]
            + effective_filter_size
            - input_size,
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.padding == "SAME":
            rows_odd, padding_rows = self._compute_padding(input, dim=0)
            cols_odd, padding_cols = self._compute_padding(input, dim=1)
            if rows_odd or cols_odd:
                input = F.pad(input, [0, cols_odd, 0, rows_odd])

            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=(padding_rows // 2, padding_cols // 2),
                dilation=self.dilation,
                groups=self.groups,
            )
