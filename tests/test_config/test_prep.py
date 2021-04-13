"""tests for vak.config.prep module"""
import vak.config.prep


def test_parse_prep_config_returns_PrepConfig_instance(
    all_generated_configs_toml_path_pairs,
):
    for config_toml, toml_path in all_generated_configs_toml_path_pairs:
        prep_section = config_toml["PREP"]
        config = vak.config.prep.PrepConfig(**prep_section)
        assert isinstance(config, vak.config.prep.PrepConfig)
