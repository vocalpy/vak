"""tests for vak.config.learncurve module"""
import copy

import pytest

import vak.config.learncurve
import vak.split


def test_learncurve_attrs_class(all_generated_learncurve_configs_toml):
    """test that instantiating LearncurveConfig class works as expected"""
    for config_toml in all_generated_learncurve_configs_toml:
        learncurve_section = config_toml['LEARNCURVE']
        config = vak.config.learncurve.LearncurveConfig(**learncurve_section)
        assert isinstance(config, vak.config.learncurve.LearncurveConfig)


def test_parse_learcurve_config_returns_attrs_class(
        all_generated_learncurve_configs_toml_path_pairs
):
    """test that ``vak.config.learncurve.parse_learncurve_config``
    returns an instance of ``vak.config.learncurve.LearncurveConfig``"""
    for config_toml, toml_path in all_generated_learncurve_configs_toml_path_pairs:
        config = vak.config.learncurve.parse_learncurve_config(
            config_toml=config_toml,
            toml_path=toml_path)
        assert isinstance(config, vak.config.learncurve.LearncurveConfig)


def test_missing_options_raises(
        all_generated_learncurve_configs_toml_path_pairs
):
    """test that configs without the required options
    in the Learncurve section raise KeyError"""
    # only need one toml/path pair so we just call next on iterator returned by fixture
    config_toml, toml_path = next(all_generated_learncurve_configs_toml_path_pairs)
    for option in vak.config.learncurve.REQUIRED_LEARNCURVE_OPTIONS:
        config_copy = copy.deepcopy(config_toml)
        config_copy['LEARNCURVE'].pop(option)
        with pytest.raises(KeyError):
            config = vak.config.learncurve.parse_learncurve_config(
                config_toml=config_copy,
                toml_path=toml_path)
