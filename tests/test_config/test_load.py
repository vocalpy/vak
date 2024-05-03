"""tests for vak.config.load module"""
import tomlkit

import vak.config.load


def test__tomlkit_to_pop(a_generated_config_path):
    with a_generated_config_path.open('r') as fp:
        tomldoc = tomlkit.load(fp)
    out = vak.config.load._tomlkit_to_popo(tomldoc)
    assert isinstance(out, dict)
    assert list(out.keys()) == ["vak"]


def test__load_from_toml_path(a_generated_config_path):
    config_dict = vak.config.load._load_toml_from_path(a_generated_config_path)

    assert isinstance(config_dict, dict)

    with a_generated_config_path.open('r') as fp:
        tomldoc = tomlkit.load(fp)
    config_dict_raw = vak.config.load._tomlkit_to_popo(tomldoc)

    assert len(list(config_dict.keys())) == len(list(config_dict_raw["vak"].keys()))
