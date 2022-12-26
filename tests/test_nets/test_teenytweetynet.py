import inspect

import torch
import pytest

import vak.nets


class TestTeenyTweetyNet:

    @pytest.mark.parametrize(
        'input_shape, num_classes',
        [
            (
                    None, 10
            ),
            (
                    None, 23
            ),
            (
                    (1, 513, 88), 6
            ),
            (
                    (1, 512, 1000), 23
            ),
        ]
    )
    def test_init(self, num_classes, input_shape):
        """test we can instantiate TeenyTweetyNet
        and it has the expected attributes"""
        if input_shape is None:
            init_sig = inspect.signature(vak.nets.TeenyTweetyNet.__init__)
            input_shape = init_sig.parameters['input_shape'].default

        net = vak.nets.TeenyTweetyNet(num_classes=num_classes, input_shape=input_shape)
        assert isinstance(net, vak.nets.TeenyTweetyNet)
        for expected_attr, expected_type in (
            ('num_classes', int),
            ('input_shape', tuple),
            ('cnn', torch.nn.Module),
            ('rnn_input_size', int),
            ('rnn', torch.nn.LSTM),
            ('fc', torch.nn.Linear)
        ):
            assert hasattr(net, expected_attr)
            assert isinstance(getattr(net, expected_attr), expected_type)

        assert net.num_classes == num_classes
        assert net.input_shape == input_shape

    @pytest.mark.parametrize(
        'input_shape, num_classes, batch_size',
        [
            (
                    None, 10, 8
            ),
            (
                    None, 23, 64
            ),
            (
                    (1, 512, 1000), 23, 64
            ),
        ]
    )
    def test_forward(self, input_shape, num_classes, batch_size):
        """test we can forward a tensor through a TeenyTweetyNet instance
        and get the expected output
        """
        if input_shape is None:
            init_sig = inspect.signature(vak.nets.TeenyTweetyNet.__init__)
            input_shape = init_sig.parameters['input_shape'].default
        input = torch.rand(batch_size, *input_shape)  # a "batch"
        net = vak.nets.TeenyTweetyNet(num_classes=num_classes)
        out = net(input)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (batch_size, num_classes, input_shape[2])
