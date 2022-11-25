"""TweetyNet model"""
from __future__ import annotations
import functools
from typing import Callable

import lightning
import torch
from torch import nn
from torch.nn import functional as F

from .. import (
    labeled_timebins,
    metrics
)


class Conv2dTF(nn.Conv2d):

    PADDING_METHODS = ('VALID', 'SAME')

    """Conv2d with padding behavior from Tensorflow

    adapted from
    https://github.com/mlperf/inference/blob/16a5661eea8f0545e04c86029362e22113c2ec09/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py#L40
    as referenced in this issue:
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-507025011

    used to maintain behavior of original implementation of TweetyNet that used Tensorflow 1.0 low-level API
    """
    def __init__(self, *args, **kwargs):
        # remove 'padding' from ``kwargs`` to avoid bug in ``torch`` => 1.7.2
        # see https://github.com/yardencsGitHub/tweetynet/issues/166
        kwargs_super = {k: v for k, v in kwargs.items() if k != 'padding'}
        super(Conv2dTF, self).__init__(*args, **kwargs_super)
        padding = kwargs.get("padding", "SAME")
        if not isinstance(padding, str):
            raise TypeError(f"value for 'padding' argument should be a string, one of: {self.PADDING_METHODS}")
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
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
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


class TweetyNet(nn.Module):
    def __init__(self,
                 num_classes,
                 input_shape=(1, 513, 88),
                 padding='SAME',
                 conv1_filters=32,
                 conv1_kernel_size=(5, 5),
                 conv2_filters=64,
                 conv2_kernel_size=(5, 5),
                 pool1_size=(8, 1),
                 pool1_stride=(8, 1),
                 pool2_size=(8, 1),
                 pool2_stride=(8, 1),
                 hidden_size=None,
                 rnn_dropout=0.,
                 num_layers=1,
                 bidirectional=True,
                 ):
        """initialize TweetyNet model

        Parameters
        ----------
        num_classes : int
            number of classes to predict, e.g., number of syllable classes in an individual bird's song
        input_shape : tuple
            with 3 elements corresponding to dimensions of spectrogram windows: (channels, frequency bins, time bins).
            i.e. we assume input is a spectrogram and treat it like an image, typically with one channel,
            the rows are frequency bins, and the columns are time bins. Default is (1, 513, 88).
        padding : str
            type of padding to use, one of {"VALID", "SAME"}. Default is "SAME".
        conv1_filters : int
            Number of filters in first convolutional layer. Default is 32.
        conv1_kernel_size : tuple
            Size of kernels, i.e. filters, in first convolutional layer. Default is (5, 5).
        conv2_filters : int
            Number of filters in second convolutional layer. Default is 64.
        conv2_kernel_size : tuple
            Size of kernels, i.e. filters, in second convolutional layer. Default is (5, 5).
        pool1_size : two element tuple of ints
            Size of sliding window for first max pooling layer. Default is (1, 8)
        pool1_stride : two element tuple of ints
            Step size for sliding window of first max pooling layer. Default is (1, 8)
        pool2_size : two element tuple of ints
            Size of sliding window for second max pooling layer. Default is (1, 8),
        pool2_stride : two element tuple of ints
            Step size for sliding window of second max pooling layer. Default is (1, 8)
        hidden_size : int
            number of features in the hidden state ``h``. Default is None,
            in which case ``hidden_size`` is set to the dimensionality of the
            output of the convolutional neural network. This default maintains
            the original behavior of the network.
        rnn_dropout : float
            If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
            with dropout probability equal to dropout. Default: 0
        num_layers : int
            Number of recurrent layers. Default is 1.
        bidirectional : bool
            If True, make LSTM bidirectional. Default is True.
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.cnn = nn.Sequential(
            Conv2dTF(in_channels=self.input_shape[0],
                     out_channels=conv1_filters,
                     kernel_size=conv1_kernel_size,
                     padding=padding
                     ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool1_size,
                         stride=pool1_stride),
            Conv2dTF(in_channels=conv1_filters,
                     out_channels=conv2_filters,
                     kernel_size=conv2_kernel_size,
                     padding=padding,
                     ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool2_size,
                         stride=pool2_stride),
        )

        # determine number of features in output after stacking channels
        # we use the same number of features for hidden states
        # note self.num_hidden is also used to reshape output of cnn in self.forward method
        batch_shape = tuple((1,) + input_shape)
        tmp_tensor = torch.rand(batch_shape)
        tmp_out = self.cnn(tmp_tensor)
        channels_out, freqbins_out = tmp_out.shape[1], tmp_out.shape[2]
        self.rnn_input_size = channels_out * freqbins_out

        if hidden_size is None:
            self.hidden_size = self.rnn_input_size
        else:
            self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size=self.rnn_input_size,
                           hidden_size=self.hidden_size,
                           num_layers=num_layers,
                           dropout=rnn_dropout,
                           bidirectional=bidirectional)

        # for self.fc, in_features = hidden_size * 2 because LSTM is bidirectional
        # so we get hidden forward + hidden backward as output
        self.fc = nn.Linear(in_features=self.hidden_size * 2, out_features=num_classes)

    def forward(self, x):
        features = self.cnn(x)
        # stack channels, to give tensor shape (batch, rnn_input_size, num time bins)
        features = features.view(features.shape[0], self.rnn_input_size, -1)
        # switch dimensions for feeding to rnn, to (num time bins, batch size, input size)
        features = features.permute(2, 0, 1)
        rnn_output, _ = self.rnn(features)
        # permute back to (batch, time bins, hidden size) to project features down onto number of classes
        rnn_output = rnn_output.permute(1, 0, 2)
        logits = self.fc(rnn_output)
        # permute yet again so that dimension order is (batch, classes, time steps)
        # because this is order that loss function expects
        return logits.permute(0, 2, 1)


class TweetyNetModel(lightning.LightningModule):
    def __init__(self,
                 network: torch.nn.Module,
                 loss_func: torch.nn.Module,
                 optimizer_config: dict,
                 lbl_tb2labels: Callable,
                 post_tfm: Callable | None = None,
                 ):
        """A neural network model that labels a time series frame by frame."""
        super().__init__()
        self.network = network
        self.loss_func = loss_func
        self.optimizer_config = optimizer_config
        self.lbl_tb2labels = lbl_tb2labels
        self.post_tfm = post_tfm

        self.metrics = {'acc': metrics.Accuracy(),
                        'levenshtein': metrics.Levenshtein(),
                        'segment_error_rate': metrics.SegmentErrorRate(),
                        'loss': loss_func}

    def forward(self, x):
        return self.network.forward(x)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.network.parameters(), **self.optimizer_config)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: rename "source" -> "spect"
        # TODO: a sample can have "spect", "audio", "annot", optionally other things ("padding"?)
        x, y = batch["source"], batch["annot"]
        # remove "batch" dimension added by collate_fn to x
        # we keep for y because loss still expects the first dimension to be batch
        # TODO: fix this weirdness. Diff't collate_fn?
        if x.ndim == 5:
            if x.shape[0] == 1:
                x = torch.squeeze(x, dim=0)
        else:
            raise ValueError(f"invalid shape for x: {x.shape}")

        out = self.network.forward(x)
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

        if self.post_tfm:
            y_pred = self.post_tfm(y_pred)

        y_labels = self.lbl_tb2labels(
            y.cpu().numpy(),
        )
        y_pred_labels = self.lbl_tb2labels(
            y_pred.cpu().numpy()
        )

        # TODO: figure out smarter way to do this
        for metric_name, metric_callable in self.metrics.items():
            if metric_name == "loss":
                self.log(f'val_{metric_name}', self.loss_func(out, y), batch_size=1)
            elif metric_name == "acc":
                self.log(f'val_{metric_name}', metric_callable(y_pred, y), batch_size=1)
            elif metric_name == "levenshtein" or metric_name == "segment_error_rate":
                self.log(f'val_{metric_name}', metric_callable(y_pred_labels, y_labels), batch_size=1)

    def predict_step(self, batch, batch_idx: int):
        x, spect_path = batch["source"].to(self.device), batch["spect_path"]
        if isinstance(spect_path, list) and len(spect_path) == 1:
            spect_path = spect_path[0]
        if x.ndim == 5:
            if x.shape[0] == 1:
                x = torch.squeeze(x, dim=0)
        y_pred = self.network.forward(x)
        return {spect_path: y_pred}

    @classmethod
    def from_config(cls, config, labelmap):
        network = TweetyNet(**config['network'])
        loss_func = torch.nn.CrossEntropyLoss(**config['loss'])
        optimizer_config = config['optimizer']
        lbl_tb2labels = functools.partial(labeled_timebins.lbl_tb2labels, labels_mapping=labelmap)
        return cls(network=network,
                   loss_func=loss_func,
                   optimizer_config=optimizer_config,
                   lbl_tb2labels=lbl_tb2labels)
