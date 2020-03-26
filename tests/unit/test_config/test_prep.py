"""tests for vak.config.prep module"""
import pytest

import vak.converters
import vak.config.prep
import vak.split


def test_parse_prep_config_returns_PrepConfig_instance(
    all_generated_configs_toml_path_pairs
):
    for config_toml, toml_path in all_generated_configs_toml_path_pairs:
        prep_config_obj = vak.config.prep.parse_prep_config(config_toml, toml_path)
        assert type(prep_config_obj) == vak.config.prep.PrepConfig


def test_no_data_dir_raises(all_generated_configs_toml_path_pairs):
    config_toml, toml_path = next(all_generated_configs_toml_path_pairs)
    config_toml['PREP'].pop('data_dir')
    with pytest.raises(KeyError):
        prep_config_obj = vak.config.prep.parse_prep_config(config_toml, toml_path)


def test_nonexistent_data_dir_raises_error(all_generated_configs_toml_path_pairs):
    config_toml, toml_path = next(all_generated_configs_toml_path_pairs)
    config_toml['PREP']['data_dir'] = 'theres/no/way/this/is/a/dir'
    with pytest.raises(NotADirectoryError):
        vak.config.prep.parse_prep_config(config_toml, toml_path)


def test_no_output_dir_raises(all_generated_configs_toml_path_pairs):
    config_toml, toml_path = next(all_generated_configs_toml_path_pairs)
    config_toml['PREP'].pop('output_dir')
    with pytest.raises(KeyError):
        prep_config_obj = vak.config.prep.parse_prep_config(config_toml, toml_path)


def test_both_audio_and_spect_format_raises(all_generated_configs_toml_path_pairs):
    """test that a config with both an audio and a spect format raises a ValueError"""
    # iterate through configs til we find one with an `audio_format` option
    # and then we'll add a `spect_format` option to it
    found_config_to_use = False
    for config_toml, toml_path in all_generated_configs_toml_path_pairs:
        if 'audio_format' in config_toml['PREP']:
            found_config_to_use = True
            break
    assert found_config_to_use  # sanity check

    config_toml['PREP']['spect_format'] = 'mat'
    with pytest.raises(ValueError):
        vak.config.prep.parse_prep_config(config_toml, toml_path)


def test_neither_audio_nor_spect_format_raises(all_generated_configs_toml_path_pairs):
    """test that a config without either an audio or a spect format raises a ValueError"""
    # iterate through configs til we find one with an `audio_format` option
    # and then we'll add a `spect_format` option to it
    found_config_to_use = False
    for config_toml, toml_path in all_generated_configs_toml_path_pairs:
        if 'audio_format' in config_toml['PREP']:
            found_config_to_use = True
            break
    assert found_config_to_use  # sanity check

    config_toml['PREP'].pop('audio_format')
    if 'spect_format' in config_toml['PREP']:
        # shouldn't be but humor me
        config_toml['PREP'].pop('spect_format')

    with pytest.raises(ValueError):
        vak.config.prep.parse_prep_config(config_toml, toml_path)
