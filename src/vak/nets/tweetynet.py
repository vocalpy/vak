"""TweetyNet model"""

from __future__ import annotations

import torch
from torch import nn

from ..nn.modules import Conv2dTF


class TweetyNet(nn.Module):
    """Neural network architecture
    that assign labels to time bins
    ("frames") in spectrogram windows.

    as described in
    https://elifesciences.org/articles/63853
    https://github.com/yardencsGitHub/tweetynet

    Cohen, Y., Nicholson, D. A., Sanchioni, A., Mallaber, E. K., Skidanova, V., & Gardner, T. J. (2022).
    Automated annotation of birdsong with a neural network that segments spectrograms. Elife, 11, e63853.

    Attributes
    ----------
    num_classes : int
        Number of classes.
        One of the two dimensions of the output.
    input_shape : tuple(int)
        With dimensions
        (channels, num. frequency bins, num. time bins in window).
    cnn : torch.nn.Sequential
        Convolutional layers of model.
    rnn_input_size : int
        Size of input to TweetyNet.rnn.
        Will be the product of the first two dimensions
        of the output of ``TweetyNet.cnn``,
        i.e. the number of output channels times
        the number of elements in the dimension
        that corresponds to frequency bins in the input.
    rnn : torch.nn.LSTM
        Bidirectional LSTM layer,
        that receives output of ``TweetyNet.cnn``.
    fc : torch.nn.Linear
        Finally fully-connected layer that maps
        the output of ``TweetyNet.rnn`` to a
        matrix of size (num. time bins in window, num. classes).

    Notes
    -----
    This is the network used by ``vak.models.TweetyNetModel``.
    """

    def __init__(
        self,
        num_classes,
        num_input_channels=1,
        num_freqbins=256,
        padding="SAME",
        conv1_filters=32,
        conv1_kernel_size=(5, 5),
        conv2_filters=64,
        conv2_kernel_size=(5, 5),
        pool1_size=(8, 1),
        pool1_stride=(8, 1),
        pool2_size=(8, 1),
        pool2_stride=(8, 1),
        hidden_size=None,
        rnn_dropout=0.0,
        num_layers=1,
        bidirectional=True,
    ):
        """initialize TweetyNet model

        Parameters
        ----------
        num_classes : int
            Number of classes to predict, e.g., number of syllable classes in an individual bird's song
        num_input_channels: int
            Number of channels in input. Typically one, for a spectrogram.
            Default is 1.
        num_freqbins: int
            Number of frequency bins in spectrograms that will be input to model.
            Default is 256.
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
        self.num_input_channels = num_input_channels
        self.num_freqbins = num_freqbins

        self.cnn = nn.Sequential(
            Conv2dTF(
                in_channels=self.num_input_channels,
                out_channels=conv1_filters,
                kernel_size=conv1_kernel_size,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool1_size, stride=pool1_stride),
            Conv2dTF(
                in_channels=conv1_filters,
                out_channels=conv2_filters,
                kernel_size=conv2_kernel_size,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool2_size, stride=pool2_stride),
        )

        # determine number of features in output after stacking channels
        # we use the same number of features for hidden states
        # note self.num_hidden is also used to reshape output of cnn in self.forward method
        N_DUMMY_TIMEBINS = (
            256  # some not-small number. This dimension doesn't matter here
        )
        batch_shape = (
            1,
            self.num_input_channels,
            self.num_freqbins,
            N_DUMMY_TIMEBINS,
        )
        tmp_tensor = torch.rand(batch_shape)
        tmp_out = self.cnn(tmp_tensor)
        channels_out, freqbins_out = tmp_out.shape[1], tmp_out.shape[2]
        self.rnn_input_size = channels_out * freqbins_out

        if hidden_size is None:
            self.hidden_size = self.rnn_input_size
        else:
            self.hidden_size = hidden_size

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )

        # for self.fc, in_features = hidden_size * 2 because LSTM is bidirectional
        # so we get hidden forward + hidden backward as output
        self.fc = nn.Linear(
            in_features=self.hidden_size * 2, out_features=num_classes
        )

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
