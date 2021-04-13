"""tests for vak.config.spect_params module"""
import pytest

import vak.config.spect_params
import vak.split

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


def test_spect_params_attrs_class(all_generated_configs_toml_path_pairs):
    """test that instantiating SpectParamsConfig class works as expected"""
    for config_toml, toml_path in all_generated_configs_toml_path_pairs:
        if "SPECT_PARAMS" in config_toml:
            spect_params_section = config_toml["SPECT_PARAMS"]
            config = vak.config.spect_params.SpectParamsConfig(**spect_params_section)
            assert isinstance(config, vak.config.spect_params.SpectParamsConfig)
