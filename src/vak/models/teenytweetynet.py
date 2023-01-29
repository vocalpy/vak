import torch
from torch import nn

from ..engine.model import Model

# absolute import to avoid name clash in model def below
import vak.metrics


class TeenyTweetyNet(nn.Module):
    def __init__(
        self,
        num_classes,
        input_shape=(1, 513, 88),
        conv1_filters=8,
        conv1_kernel_size=(5, 5),
        conv1_padding=(0, 2),
        conv2_filters=16,
        conv2_kernel_size=(5, 5),
        conv2_padding=(0, 2),
        pool1_size=(4, 1),
        pool1_stride=(4, 1),
        pool2_size=(4, 1),
        pool2_stride=(4, 1),
        hidden_size=32,
    ):
        """TeenyTweetyNet model

        Parameters
        ----------
        num_classes : int
            number of classes to predict, e.g., number of syllable classes in an individual bird's song
        input_shape : tuple
            with 3 elements corresponding to dimensions of spectrogram windows: (channels, frequency bins, time bins).
            i.e. we assume input is a spectrogram and treat it like an image, typically with one channel,
            the rows are frequency bins, and the columns are time bins. Default is (1, 513, 88).
        conv1_filters : int
            Number of filters in first convolutional layer. Default is 32.
        conv1_kernel_size : tuple
            Size of kernels, i.e. filters, in first convolutional layer. Default is (5, 5).
        conv2_filters : int
            Number of filters in second convolutional layer. Default is 64.
        conv2_kernel_size : tuple
            Size of kernels, i.e. filters, in second convolutional layer. Default is (5, 5).
        pool1_size : two element tuple of ints
            Size of sliding window for first max pooling layer. Default is (8, 1)
        pool1_stride : two element tuple of ints
            Step size for sliding window of first max pooling layer. Default is (8, 1)
        pool2_size : two element tuple of ints
            Size of sliding window for second max pooling layer. Default is (4, 1),
        pool2_stride : two element tuple of ints
            Step size for sliding window of second max pooling layer. Default is (4, 1)
        hidden_size : int
            Size of hidden state in recurrent neural network; dimensionality of vector.
            Default is 32.
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.hidden_size = hidden_size

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_shape[0],
                out_channels=conv1_filters,
                kernel_size=conv1_kernel_size,
                padding=conv1_padding,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool1_size, stride=pool1_stride),
            nn.Conv2d(
                in_channels=conv1_filters,
                out_channels=conv2_filters,
                kernel_size=conv2_kernel_size,
                padding=conv2_padding,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool2_size, stride=pool2_stride),
        )

        # determine number of features in output after stacking channels
        # we use the same number of features for hidden states
        # note self.num_hidden is also used to reshape output of cnn in self.forward method
        batch_shape = tuple((1,) + input_shape)
        tmp_tensor = torch.rand(batch_shape)
        tmp_out = self.cnn(tmp_tensor)
        channels_out, freqbins_out = tmp_out.shape[1], tmp_out.shape[2]
        self.num_rnn_features = channels_out * freqbins_out

        self.rnn = nn.LSTM(
            input_size=self.num_rnn_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=0,
            bidirectional=True,
        )

        # for self.fc, in_features = hidden_size * 2, because LSTM is bidirectional
        # so we get hidden forward + hidden backward as output
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        # stack channels so that dimension order is (batch, num_rnn_features, num time bins)
        features = features.view(features.shape[0], self.num_rnn_features, -1)
        # now switch dimensions for feeding to rnn,
        # so dimension order is (num time bins, batch size, num_rnn_features)
        features = features.permute(2, 0, 1)
        rnn_output, (hidden, cell_state) = self.rnn(features)
        # permute back to (batch, time bins, features)
        # so we can project features down onto number of classes
        rnn_output = rnn_output.permute(1, 0, 2)
        logits = self.fc(rnn_output)
        # permute yet again
        # so that dimension order is (batch, classes, time steps)
        # because this is order that loss function expects
        return logits.permute(0, 2, 1)


class TeenyTweetyNetModel(Model):
    @classmethod
    def from_config(cls, config, post_tfm=None):
        network = TeenyTweetyNet(**config["network"])
        loss = nn.CrossEntropyLoss(**config["loss"])
        optimizer = torch.optim.Adam(params=network.parameters(), **config["optimizer"])
        metrics = {
            "acc": vak.metrics.Accuracy(),
            "levenshtein": vak.metrics.Levenshtein(),
            "segment_error_rate": vak.metrics.SegmentErrorRate(),
            "loss": torch.nn.CrossEntropyLoss(),
        }
        return cls(
            network=network,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            post_tfm=post_tfm
        )
