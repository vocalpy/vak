"""tests for vak.config.spect_params module"""
import pytest

import vak.config.spect_params

from .attr_helpers import simple_attr


# approaching to testing validators adopted from
# https://github.com/python-attrs/attrs/blob/a025629e36440dcc27aee0ee5b04d6523bcc9931/tests/test_validators.py#L338
def test_freq_cutoffs_not_two_values_raises():
    """test that a ValueError is raised when freq_cutoffs is not just two values"""
    freq_cutoff_attrib = simple_attr("freq_cutoffs")

    with pytest.raises(ValueError):
        vak.config.spect_params.freq_cutoffs_validator(
            instance=None, attribute=freq_cutoff_attrib, value=[0]
        )

    with pytest.raises(ValueError):
        vak.config.spect_params.freq_cutoffs_validator(
            instance=None, attribute=freq_cutoff_attrib, value=[0, 10, 100]
        )


def test_invalid_transform_type_raises():
    """test that an invalid spectrogram transform type raises a ValueError"""
    transform_type_attrib = simple_attr("transform_type")

    with pytest.raises(ValueError):
        vak.config.spect_params.is_valid_transform_type(
            instance=None,
            attribute=transform_type_attrib,
            value="log",  # not in vak.config.spect_params.VALID_TRANSFORM_TYPES
        )


def test_freq_cutoffs_wrong_order_raises():
    """test that a ValueError is raised when freq_cutoffs[0] > freq_cutoffs[1]"""
    freq_cutoff_attrib = simple_attr("freq_cutoffs")

    with pytest.raises(ValueError):
        vak.config.spect_params.freq_cutoffs_validator(
            instance=None, attribute=freq_cutoff_attrib, value=[10000, 500]
        )


class TestSpectParamsConfig:
    @pytest.mark.parametrize(
        'config_dict',
        [
            {'fft_size': 512, 'step_size': 64, 'freq_cutoffs': [500, 10000], 'thresh': 6.25, 'transform_type': 'log_spect'},
        ]
    )
    def test_init(self, config_dict):
        spect_params_config = vak.config.SpectParamsConfig(**config_dict)
        assert isinstance(spect_params_config, vak.config.spect_params.SpectParamsConfig)

    def test_with_real_config(self, a_generated_config_dict):
        if "spect_params" in a_generated_config_dict['prep']:
            spect_config_dict = a_generated_config_dict['prep']['spect_params']
        else:
            pytest.skip("No spect params in config")
        spect_params_config = vak.config.spect_params.SpectParamsConfig(**spect_config_dict)
        assert isinstance(spect_params_config, vak.config.spect_params.SpectParamsConfig)
