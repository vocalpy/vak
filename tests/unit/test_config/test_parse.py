"""tests for vak.config.parse module"""
import pytest

import vak.config
import vak.transforms.transforms
import vak.split
import vak.models
import vak.spect


def test_config_attrs_class(
        all_generated_configs_toml_path_pairs
):
    """test that instantiating Config class works as expected"""
    for config_toml, toml_path in all_generated_configs_toml_path_pairs:
    # this test is basically the body of the ``config.parse.from_toml`` function.
        config_dict = {}
        for section_name in vak.config.parse.SECTION_PARSERS.keys():
            if section_name in config_toml:
                vak.config.validators.are_options_valid(config_toml, section_name, toml_path)
                section_parser = vak.config.parse.SECTION_PARSERS[section_name]
                config_dict[section_name.lower()] = section_parser(config_toml, toml_path)

        config = vak.config.parse.Config(**config_dict)
        assert isinstance(config, vak.config.parse.Config)


def test_from_toml_path_returns_instance_of_config(all_generated_configs):
    for toml_path in all_generated_configs:
        config_obj = vak.config.parse.from_toml_path(toml_path)
        assert isinstance(config_obj, vak.config.parse.Config)


def test_invalid_section_raises(invalid_section_config_path):
    with pytest.raises(ValueError):
        vak.config.parse.from_toml_path(invalid_section_config_path)


def test_invalid_option_raises(invalid_option_config_path):
    with pytest.raises(ValueError):
        vak.config.parse.from_toml_path(invalid_option_config_path)


@pytest.fixture
def invalid_train_and_learncurve_config_toml(test_configs_root):
    return test_configs_root.joinpath('invalid_train_and_learncurve_config.toml')


def test_train_and_learncurve_defined_raises(invalid_train_and_learncurve_config_toml):
    """test that a .toml config with both a TRAIN and a LEARNCURVE section raises a ValueError"""
    with pytest.raises(ValueError):
        vak.config.parse.from_toml_path(invalid_train_and_learncurve_config_toml)
