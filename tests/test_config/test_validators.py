import pytest
import toml

import vak.config.validators


def test_are_sections_valid(invalid_section_config_path):
    """test that invalid section name raises a ValueError"""
    with invalid_section_config_path.open("r") as fp:
        config_toml = toml.load(fp)
    with pytest.raises(ValueError):
        vak.config.validators.are_sections_valid(
            config_toml, invalid_section_config_path
        )


def test_are_options_valid(invalid_option_config_path):
    """test that section with an invalid option name raises a ValueError"""
    section_with_invalid_option = "PREP"
    with invalid_option_config_path.open("r") as fp:
        config_toml = toml.load(fp)
    with pytest.raises(ValueError):
        vak.config.validators.are_options_valid(
            config_toml, section_with_invalid_option, invalid_option_config_path
        )
