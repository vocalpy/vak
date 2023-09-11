import itertools

import pytest
import pytorch_lightning as lightning

import vak


LABELSETS = []
for labelset in (
    'abcde',
    '12345',
    [1, 2, 3, 4, 5],
    'mo', 'po', 'ta',
):
    LABELSETS.append(
        vak.common.converters.labelset_to_set(labelset)
    )
LABELMAPS = []
for labelset in LABELSETS:
    LABELMAPS.append(
        vak.common.labels.to_map(labelset, map_unlabeled=True)
    )

INPUT_SHAPES = (
    (1, 513, 88),
    (1, 267, 1000)
)

TEST_INIT_ARGVALS = itertools.product(LABELMAPS, INPUT_SHAPES)


class TestTweetyNet:
    def test_model_is_decorated(self):
        assert issubclass(vak.models.TweetyNet,
                          vak.models.FrameClassificationModel)
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
        num_input_channels = input_shape[-3]
        num_freqbins = input_shape[-2]
        network = vak.models.TweetyNet.definition.network(len(labelmap), num_input_channels, num_freqbins)
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
