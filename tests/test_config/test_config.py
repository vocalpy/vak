import vak.config


class TestConfig:
    def test_from_config_dict_with_real_config(
        a_generated_config_dict,
    ):
        """test that instantiating Config class works as expected"""
        # this is basically the body of the ``config.load.from_toml`` function.
        config_kwargs = {}
        for table_name in a_generated_config_dict:
            config_kwargs[table_name] = vak.config.load.TABLE_CLASSES_MAP[table_name].from_config_dict(
                a_generated_config_dict[table_name]
            )

        config = vak.config.load.Config(**config_kwargs)

        assert isinstance(config, vak.config.load.Config)
        # we already test that config loading works for EvalConfig, et al.,
        # so here we just test that the logic of Config works as expected:
        # we should get an attribute for each top-level table that we pass in;
        # if we don't pass one in, then its corresponding attribute should be None
        for table_name in ('eval', 'learncurve', 'predict', 'prep', 'train'):
            if table_name in a_generated_config_dict:
                assert hasattr(config, table_name)
            else:
                assert getattr(config, table_name) is None
