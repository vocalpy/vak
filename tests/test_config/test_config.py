import vak.config


def test_config_attrs_class(
    all_generated_configs_toml_path_pairs,
    default_model,
):
    """test that instantiating Config class works as expected"""
    for config_toml, toml_path in all_generated_configs_toml_path_pairs:
        if default_model not in str(toml_path):
            continue  # only need to check configs for one model
            # also avoids FileNotFoundError on CI
        # this is basically the body of the ``config.parse.from_toml`` function.
        config_dict = {}
        for section_name in list(vak.config.parse.SECTION_CLASSES.keys()):
            if section_name in config_toml:
                vak.config.validators.are_options_valid(
                    config_toml, section_name, toml_path
                )
                section = vak.config.parse.parse_config_section(
                    config_toml, section_name, toml_path
                )
                config_dict[section_name.lower()] = section

        config = vak.config.parse.Config(**config_dict)
        assert isinstance(config, vak.config.parse.Config)
