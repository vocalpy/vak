import inspect

import torch
import pytest

import vak.nets


class TestTweetyNet:

    @pytest.mark.parametrize(
        'num_classes, num_input_channels, num_freqbins',
        [
            (
                    1, 10, None,
            ),
            (
                    1, 23, None,
            ),
            (
                    6, 1, 513
            ),
            (
                    23, 1, 512
            ),
        ]
    )
    def test_init(self, num_classes, num_input_channels, num_freqbins):
        """test we can instantiate TweetyNet
        and it has the expected attributes"""
        if num_input_channels is None or num_freqbins is None:
            init_sig = inspect.signature(vak.nets.TweetyNet.__init__)
            if num_input_channels is None:
                num_input_channels = init_sig.parameters['num_input_channels'].default
            if num_freqbins is None:
                num_freqbins = init_sig.parameters['num_freqbins'].default

        net = vak.nets.TweetyNet(num_classes, num_input_channels, num_freqbins)
        assert isinstance(net, vak.nets.TweetyNet)
        for expected_attr, expected_type in (
            ('num_classes', int),
            ('num_input_channels', int),
            ('num_freqbins', int),
            ('cnn', torch.nn.Module),
            ('rnn_input_size', int),
            ('rnn', torch.nn.LSTM),
            ('fc', torch.nn.Linear)
        ):
            assert hasattr(net, expected_attr)
            assert isinstance(getattr(net, expected_attr), expected_type)

        assert net.num_classes == num_classes
        assert net.num_input_channels == num_input_channels
        assert net.num_freqbins == num_freqbins

    @pytest.mark.parametrize(
        'num_classes, num_input_channels, num_freqbins, num_timebins, batch_size',
        [
            (
                    10, None, None, 100, 8
            ),
            (
                    23, None, None, 100, 64
            ),
            (
                    23, 1, 512, 100, 64
            ),
        ]
    )
    def test_forward(self, num_classes, num_input_channels, num_freqbins, num_timebins, batch_size):
        """test we can forward a tensor through a TweetyNet instance
        and get the expected output"""
        if num_input_channels is None or num_freqbins is None:
            init_sig = inspect.signature(vak.nets.TweetyNet.__init__)
            if num_input_channels is None:
                num_input_channels = init_sig.parameters['num_input_channels'].default
            if num_freqbins is None:
                num_freqbins = init_sig.parameters['num_freqbins'].default

        input = torch.rand(batch_size, num_input_channels, num_freqbins, num_timebins)  # a "batch"
        net = vak.nets.TweetyNet(num_classes, num_input_channels, num_freqbins)
        out = net(input)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (batch_size, num_classes, num_timebins)
