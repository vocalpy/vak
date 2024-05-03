import pytest

import vak.config.validators


def test_are_tables_valid(invalid_table_config_path):
    """test that invalid table name raises a ValueError"""
    config_dict = vak.config.load._load_toml_from_path(invalid_table_config_path)
    with pytest.raises(ValueError):
        vak.config.validators.are_tables_valid(
            config_dict, invalid_table_config_path
        )


def test_are_keys_valid(invalid_key_config_path):
    """test that table with an invalid key name raises a ValueError"""
    table_with_invalid_key = "prep"
    config_dict = vak.config.load._load_toml_from_path(invalid_key_config_path)
    with pytest.raises(ValueError):
        vak.config.validators.are_keys_valid(
            config_dict, table_with_invalid_key, invalid_key_config_path
        )
