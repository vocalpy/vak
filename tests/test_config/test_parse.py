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


def test_load_from_toml_path(all_generated_configs):
    for toml_path in all_generated_configs:
        config_toml = vak.config.parse._load_toml_from_path(toml_path)
        assert isinstance(config_toml, dict)


def test_load_from_toml_path_raises_when_config_doesnt_exist(config_that_doesnt_exist):
    with pytest.raises(FileNotFoundError):
        vak.config.parse._load_toml_from_path(config_that_doesnt_exist)


def test_from_toml_path_returns_instance_of_config(all_generated_configs):
    for toml_path in all_generated_configs:
        config_obj = vak.config.parse.from_toml_path(toml_path)
        assert isinstance(config_obj, vak.config.parse.Config)


def test_from_toml_path_raises_when_config_doesnt_exist(config_that_doesnt_exist):
    with pytest.raises(FileNotFoundError):
        vak.config.parse.from_toml_path(config_that_doesnt_exist)


def test_from_toml(all_generated_configs_toml_path_pairs):
    for config_toml, toml_path in all_generated_configs_toml_path_pairs:
        config_obj = vak.config.parse.from_toml(config_toml, toml_path)
        assert isinstance(config_obj, vak.config.parse.Config)


def test_from_toml_with_sections_not_none(all_generated_configs_toml_path_pairs):
    for config_toml, toml_path in all_generated_configs_toml_path_pairs:
        config_obj = vak.config.parse.from_toml(config_toml, toml_path, sections=['PREP', 'SPECT_PARAMS'])
        assert isinstance(config_obj, vak.config.parse.Config)
        for should_have in ('prep', 'spect_params'):
            assert hasattr(config_obj, should_have)
        for should_be_none in ('eval', 'learncurve', 'train', 'predict'):
            assert getattr(config_obj, should_be_none) is None
        assert getattr(config_obj, 'dataloader') == vak.config.dataloader.DataLoaderConfig()


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
