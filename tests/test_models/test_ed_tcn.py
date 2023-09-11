import pytest

import vak

from .test_tweetynet import TEST_INIT_ARGVALS


class TestED_TCN:
    @pytest.mark.parametrize(
        'labelmap, input_shape',
        TEST_INIT_ARGVALS
    )
    def test_init(self, labelmap, input_shape):
        # network has required args that need to be determined dynamically
        num_input_channels = input_shape[-3]
        num_freqbins = input_shape[-2]
        network = vak.models.ED_TCN.definition.network(len(labelmap), num_input_channels, num_freqbins)
        model = vak.models.ED_TCN(labelmap=labelmap, network=network)
        assert isinstance(model, vak.models.ED_TCN)
        for attr in ('network', 'loss', 'optimizer'):
            assert hasattr(model, attr)
            assert isinstance(getattr(model, attr),
                              getattr(vak.models.ed_tcn.ED_TCN.definition, attr))
        assert hasattr(model, 'metrics')
        assert isinstance(model.metrics, dict)
        for metric_name, metric_callable in model.metrics.items():
            assert isinstance(metric_callable,
                              vak.models.ed_tcn.ED_TCN.definition.metrics[metric_name])
