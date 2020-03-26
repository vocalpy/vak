"""tests for vak.config.train module"""
import copy

import pytest

import vak.config.train
import vak.split


def test_parse_train_config_returns_TrainConfig_instance(
        all_generated_train_configs_toml_path_pairs
):
    """test that instantiating TrainConfig class works as expected"""
    for config_toml, toml_path in all_generated_train_configs_toml_path_pairs:
        train_config_obj = vak.config.train.parse_train_config(config_toml, toml_path)
        assert isinstance(train_config_obj, vak.config.train.TrainConfig)


def test_missing_options_raises(
        all_generated_train_configs_toml_path_pairs
):
    """test that configs without the required options
    in the Train section raise KeyError"""
    # only need one toml/path pair so we just call next on iterator returned by fixture
    config_toml, toml_path = next(all_generated_train_configs_toml_path_pairs)
    for option in vak.config.train.REQUIRED_TRAIN_OPTIONS:
        config_copy = copy.deepcopy(config_toml)
        config_copy['TRAIN'].pop(option)
        with pytest.raises(KeyError):
            config = vak.config.learncurve.parse_learncurve_config(
                config_toml=config_copy,
                toml_path=toml_path)


def test_invalid_model_name_raises(all_generated_train_configs_toml_path_pairs):
    # only need one toml/path pair so we just call next on iterator returned by fixture
    config_toml, toml_path = next(all_generated_train_configs_toml_path_pairs)
    config_toml['TRAIN']['models'] = ['NotInstalledNet', 'OtherNotInstalledNet']
    with pytest.raises(ValueError):
        vak.config.train.parse_train_config(config_toml, toml_path)


def test_nonexistent_root_results_dir_raises(
        all_generated_train_configs_toml_path_pairs
):

    # only need one toml/path pair so we just call next on iterator returned by fixture
    config_toml, toml_path = next(all_generated_train_configs_toml_path_pairs)
    config_toml['TRAIN']['root_results_dir'] = 'obviously/non/existent/dir'
    with pytest.raises(NotADirectoryError):
        vak.config.train.parse_train_config(config_toml, toml_path)
