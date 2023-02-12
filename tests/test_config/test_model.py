import copy
import pytest

from ..fixtures import (
    ALL_GENERATED_CONFIGS_TOML,
    ALL_GENERATED_CONFIGS_TOML_PATH_PAIRS
)

import vak.config.model


def _make_expected_config(model_config: dict) -> dict:
    for attr in vak.config.model.MODEL_TABLES:
        if attr not in model_config:
            model_config[attr] = {}
    return model_config


@pytest.mark.parametrize(
    'toml_dict',
    ALL_GENERATED_CONFIGS_TOML
)
def test_config_from_toml_dict(toml_dict):
    for section_name in ('TRAIN', 'EVAL', 'LEARNCURVE', 'PREDICT'):
        try:
            section = toml_dict[section_name]
        except KeyError:
            continue
    model_name = section['model']
    # we need to copy so that we don't silently fail to detect mistakes
    # by comparing a reference to the dict with itself
    expected_model_config = copy.deepcopy(
        toml_dict[model_name]
    )
    expected_model_config = _make_expected_config(expected_model_config)

    model_config = vak.config.model.config_from_toml_dict(toml_dict, model_name)

    assert model_config == expected_model_config


@pytest.mark.parametrize(
    'toml_dict, toml_path',
    ALL_GENERATED_CONFIGS_TOML_PATH_PAIRS
)
def test_config_from_toml_path(toml_dict, toml_path):
    for section_name in ('TRAIN', 'EVAL', 'LEARNCURVE', 'PREDICT'):
        try:
            section = toml_dict[section_name]
        except KeyError:
            continue
    model_name = section['model']
    # we need to copy so that we don't silently fail to detect mistakes
    # by comparing a reference to the dict with itself
    expected_model_config = copy.deepcopy(
        toml_dict[model_name]
    )
    expected_model_config = _make_expected_config(expected_model_config)

    model_config = vak.config.model.config_from_toml_path(toml_path, model_name)

    assert model_config == expected_model_config
