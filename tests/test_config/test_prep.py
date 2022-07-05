"""tests for vak.config.prep module"""
import vak.config.prep


def test_parse_prep_config_returns_PrepConfig_instance(
        configs_toml_path_pairs_by_model_factory,
        model,
):
    config_toml_path_pairs = configs_toml_path_pairs_by_model_factory(model)
    for config_toml, toml_path in config_toml_path_pairs:
        prep_section = config_toml["PREP"]
        config = vak.config.prep.PrepConfig(**prep_section)
        assert isinstance(config, vak.config.prep.PrepConfig)
