"""tests for vak.config.predict module"""
import copy

import pytest

import vak.config.predict
import vak.split


def test_predict_attrs_class(
        all_generated_predict_configs_toml
):
    """test that instantiating PredictConfig class works as expected"""
    for config_toml in all_generated_predict_configs_toml:
        predict_section = config_toml['PREDICT']
        config = vak.config.predict.PredictConfig(**predict_section)
        assert isinstance(config, vak.config.predict.PredictConfig)


def test_parse_predict_config_returns_PredictConfig_instance(
        all_generated_predict_configs_toml_path_pairs
):
    for config_toml, toml_path in all_generated_predict_configs_toml_path_pairs:
        predict_config_obj = vak.config.predict.parse_predict_config(
            config_toml=config_toml, toml_path=toml_path
        )
        assert isinstance(predict_config_obj, vak.config.predict.PredictConfig)


def test_no_models_option_raises(
        all_generated_predict_configs_toml_path_pairs
):
    """test that configs without the required options
        in the PREDICT section raise KeyError"""
    # we only need one toml, path pair
    # so we just call next on the ``zipped`` iterator that our fixture gives us
    config_toml, toml_path = next(all_generated_predict_configs_toml_path_pairs)
    for option in vak.config.predict.REQUIRED_PREDICT_OPTIONS:
        config_copy = copy.deepcopy(config_toml)
        config_copy['PREDICT'].pop(option)
        with pytest.raises(KeyError):
            config = vak.config.predict.parse_predict_config(
                config_toml=config_copy,
                toml_path=toml_path
            )


def test_model_not_installed_raises(
        all_generated_predict_configs_toml_path_pairs
):
    """test that a ValueError is raised when the ``models`` option
    in the PREDICT section specifies names of models that are not installed"""
    # we only need one toml, path pair
    # so we just call next on the ``zipped`` iterator that our fixture gives us
    config_toml, toml_path = next(all_generated_predict_configs_toml_path_pairs)
    config_toml['PREDICT']['models'] = 'NotInstalledNet, OtherNotInstalledNet'
    with pytest.raises(ValueError):
        vak.config.predict.parse_predict_config(
            config_toml=config_toml,
            toml_path=toml_path
        )


def test_nonexistent_checkpoint_path_raises(
        all_generated_predict_configs_toml_path_pairs
):
    """test that a FileNotFoundError is raised when checkpoint_path does not exist"""
    # we only need one toml, path pair
    # so we just call next on the ``zipped`` iterator that our fixture gives us
    config_toml, toml_path = next(all_generated_predict_configs_toml_path_pairs)
    config_toml['PREDICT']['checkpoint_path'] = 'obviously/non/existent/dir/check.pt'
    with pytest.raises(FileNotFoundError):
        vak.config.predict.parse_predict_config(
            config_toml=config_toml,
            toml_path=toml_path
        )


def test_nonexistent_csv_path_raises(
        all_generated_predict_configs_toml_path_pairs
):
    """test that a FileNotFoundError is raised when csv_path does not exist"""
    # we only need one toml, path pair
    # so we just call next on the ``zipped`` iterator that our fixture gives us
    config_toml, toml_path = next(all_generated_predict_configs_toml_path_pairs)
    config_toml['PREDICT']['csv_path'] = 'obviously/non/existent/dir/predict.csv'
    with pytest.raises(FileNotFoundError):
        vak.config.predict.parse_predict_config(
            config_toml=config_toml,
            toml_path=toml_path
        )
