import pytest

import vak


class TestConvEncoderUMAP:
    @pytest.mark.parametrize(
        'input_shape',
        [
            (1, 32, 32),
            (1, 64, 64),
        ]
    )
    def test_init(self, input_shape):
        network = {
            'encoder': vak.models.ConvEncoderUMAP.definition.network['encoder'](input_shape=input_shape)
        }
        model = vak.models.ConvEncoderUMAP(network=network)
        assert isinstance(model, vak.models.ConvEncoderUMAP)
        for attr in ('network', 'loss', 'optimizer'):
            assert hasattr(model, attr)
            attr_from_definition = getattr(vak.models.convencoder_umap.ConvEncoderUMAP.definition, attr)
            if isinstance(attr_from_definition, dict):
                attr_from_model = getattr(model, attr)
                assert isinstance(attr_from_model, dict)
                assert attr_from_model.keys() == attr_from_definition.keys()
                for net_name, net_instance in attr_from_model.items():
                    assert isinstance(net_instance, attr_from_definition[net_name])
            else:
                assert isinstance(getattr(model, attr),
                                  getattr(vak.models.convencoder_umap.ConvEncoderUMAP.definition, attr))
        assert hasattr(model, 'metrics')
        assert isinstance(model.metrics, dict)
        for metric_name, metric_callable in model.metrics.items():
            assert isinstance(metric_callable,
                              vak.models.convencoder_umap.ConvEncoderUMAP.definition.metrics[metric_name])
