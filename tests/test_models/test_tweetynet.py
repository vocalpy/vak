import itertools
import sys

import pytest
import pytorch_lightning as lightning

import vak.models


LABELSETS = []
for labelset in (
    'abcde',
    '12345',
    [1, 2, 3, 4, 5],
    'mo', 'po', 'ta',
):
    LABELSETS.append(
        ['unlabeled'] + list(labelset)
    )
LABELMAPS = []
for labelset in LABELSETS:
    LABELMAPS.append(
        dict(zip(labelset, range(len(labelset))))
    )

INPUT_SHAPES = (
    (1, 513, 88),
    (1, 267, 1000)
)

TEST_INIT_ARGVALS = itertools.product(LABELMAPS, INPUT_SHAPES)


class TestTweetyNet:
    def test_model_is_decorated(self):
        assert issubclass(vak.models.TweetyNet,
                          vak.models.WindowedFrameClassificationModel)
        assert issubclass(vak.models.TweetyNet,
                          vak.models.base.Model)
        assert issubclass(vak.models.TweetyNet,
                          lightning.LightningModule)

    @pytest.mark.parametrize(
        'labelmap, input_shape',
        TEST_INIT_ARGVALS
    )
    def test_init(self, labelmap, input_shape):
        # network has required args that need to be determined dynamically
        network = vak.models.TweetyNet.definition.network(num_classes=len(labelmap),
                                                          input_shape=input_shape)
        model = vak.models.TweetyNet(labelmap=labelmap, network=network)
        assert isinstance(model, vak.models.TweetyNet)
        for attr in ('network', 'loss', 'optimizer'):
            assert hasattr(model, attr)
            assert isinstance(getattr(model, attr),
                              getattr(vak.models.tweetynet.TweetyNet.definition, attr))
        assert hasattr(model, 'metrics')
        assert isinstance(model.metrics, dict)
        for metric_name, metric_callable in model.metrics.items():
            assert isinstance(metric_callable,
                              vak.models.tweetynet.TweetyNet.definition.metrics[metric_name])
