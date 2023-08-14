import torch

from ..nn.modules import Conv2dTF, NormReLU


class ED_TCN(torch.nn.Module):
    """Encoder-Decoder Temporal Convolutional Network.
    As described in [1]_.

    Note that this network adds convolutional layers on the front end
    to provide features fed into the ED-TCN described in [1]_.

    This implementation in PyTorch is adapted from the original in Keras
    under MIT license.
    https://github.com/colincsl/TemporalConvolutionalNetworks/blob/cccdf868ed4c7a56745b41d30fd9f1dc637eb3f3/code/tf_models.py#L68
    https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/LICENSE

    .. [1] Lea, C., Flynn, M. D., Vidal, R., Reiter, A., & Hager, G. D. (2017).
       Temporal convolutional networks for action segmentation and detection.
       In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 156-165).
    """

    def __init__(
        self,
        num_classes,
        num_input_channels=1,
        num_freqbins=256,
        padding="SAME",
        conv2d_1_filters=32,
        conv2d_1_kernel_size=(5, 5),
        conv2d_2_filters=64,
        conv2d_2_kernel_size=(5, 5),
        pool1_size=(8, 1),
        pool1_stride=(8, 1),
        pool2_size=(8, 1),
        pool2_stride=(8, 1),
        conv1d_1_filters=64,
        conv1d_2_filters=96,
        conv1d_kernel_size=25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.num_freqbins = num_freqbins

        self.cnn = torch.nn.Sequential(
            Conv2dTF(
                in_channels=self.num_input_channels,
                out_channels=conv2d_1_filters,
                kernel_size=conv2d_1_kernel_size,
                padding=padding,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=pool1_size, stride=pool1_stride),
            Conv2dTF(
                in_channels=conv2d_1_filters,
                out_channels=conv2d_2_filters,
                kernel_size=conv2d_2_kernel_size,
                padding=padding,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=pool2_size, stride=pool2_stride),
        )

        # determine number of features in output after stacking channels
        # we use the same number of features for hidden states
        # note self.num_hidden is also used to reshape output of cnn in self.forward method
        # determine number of features in output after stacking channels
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
        self.n_cnn_features_out = channels_out * freqbins_out

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.n_cnn_features_out,
                conv1d_1_filters,
                conv1d_kernel_size,
                padding="same",
            ),
            torch.nn.Dropout1d(p=0.3),
            NormReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(
                conv1d_1_filters,
                conv1d_2_filters,
                conv1d_kernel_size,
                padding="same",
            ),
            torch.nn.Dropout1d(0.3),
            NormReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv1d(
                conv1d_2_filters,
                conv1d_2_filters,
                conv1d_kernel_size,
                padding="same",
            ),
            torch.nn.Dropout1d(p=0.3),
            NormReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv1d(
                conv1d_2_filters,
                conv1d_1_filters,
                conv1d_kernel_size,
                padding="same",
            ),
            torch.nn.Dropout1d(0.3),
            NormReLU(),
        )

        self.fc = torch.nn.Linear(
            in_features=conv1d_1_filters, out_features=self.num_classes
        )

    def forward(self, x):
        x = self.cnn(x)
        # stack channels, to give tensor shape (batch, cnn features, time bins)
        x = x.view(x.shape[0], self.n_cnn_features_out, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(
            0, 2, 1
        )  # so that we can project features down on to number of classes
        x = self.fc(x)
        x = x.permute(
            0, 2, 1
        )  # switch back to (batch, classes, time) for loss function
        return x
