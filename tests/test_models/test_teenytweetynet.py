import pytest
import pytorch_lightning as lightning

import vak.models

from .test_tweetynet import TEST_INIT_ARGVALS


class TestTeenyTweetyNet:
    def test_model_is_decorated(self):
        assert issubclass(vak.models.TeenyTweetyNet,
                          vak.models.FrameClassificationModel)
        assert issubclass(vak.models.TeenyTweetyNet,
                          vak.models.base.Model)
        assert issubclass(vak.models.TeenyTweetyNet,
                          lightning.LightningModule)

    @pytest.mark.parametrize(
        'labelmap, input_shape',
        TEST_INIT_ARGVALS
    )
    def test_init(self, labelmap, input_shape):
        # network has required args that need to be determined dynamically
        num_input_channels = input_shape[-3]
        num_freqbins = input_shape[-2]
        network = vak.models.TeenyTweetyNet.definition.network(len(labelmap), num_input_channels, num_freqbins)
        model = vak.models.TeenyTweetyNet(labelmap=labelmap, network=network)
        assert isinstance(model, vak.models.TeenyTweetyNet)
        for attr in ('network', 'loss', 'optimizer'):
            assert hasattr(model, attr)
            assert isinstance(getattr(model, attr),
                              getattr(vak.models.teenytweetynet.TeenyTweetyNet.definition, attr))
        assert hasattr(model, 'metrics')
        assert isinstance(model.metrics, dict)
        for metric_name, metric_callable in model.metrics.items():
            assert isinstance(metric_callable,
                              vak.models.teenytweetynet.TeenyTweetyNet.definition.metrics[metric_name])
