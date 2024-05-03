import pytest

import vak.config


class TestConfig:
    def test_from_config_dict_with_real_config(
        self, a_generated_config_dict,
    ):
        """test that instantiating Config class works as expected"""
        # this is basically the body of the ``config.load.from_toml`` function.
        config_kwargs = {}
        for table_name in a_generated_config_dict:
            config_kwargs[table_name] = vak.config.config.TABLE_CLASSES_MAP[table_name].from_config_dict(
                a_generated_config_dict[table_name]
            )

        config = vak.config.Config(**config_kwargs)

        assert isinstance(config, vak.config.Config)
        # we already test that config loading works for EvalConfig, et al.,
        # so here we just test that the logic of Config works as expected:
        # we should get an attribute for each top-level table that we pass in;
        # if we don't pass one in, then its corresponding attribute should be None
        for table_name in ('eval', 'learncurve', 'predict', 'prep', 'train'):
            if table_name in a_generated_config_dict:
                assert hasattr(config, table_name)
            else:
                assert getattr(config, table_name) is None

    def test_from_toml_path(self, a_generated_config_path):
        config_toml = vak.config.load._load_toml_from_path(a_generated_config_path)
        assert isinstance(config_toml, dict)

    def test_from_toml_path_raises_when_config_doesnt_exist(self, config_that_doesnt_exist):
        with pytest.raises(FileNotFoundError):
            vak.config.Config.from_toml_path(config_that_doesnt_exist)

    def test_invalid_table_raises(self, invalid_table_config_path):
        with pytest.raises(ValueError):
            vak.config.Config.from_toml_path(invalid_table_config_path)

    def test_invalid_key_raises(self, invalid_key_config_path):
        with pytest.raises(ValueError):
            vak.config.Config.from_toml_path(invalid_key_config_path)

    def test_mutiple_top_level_tables_besides_prep_raises(self, invalid_train_and_learncurve_config_toml):
        """Test that a .toml config with two top-level tables besides ``[vak.prep]`` raises a ValueError
        (in this case ``[vak.train]`` and ``[vak.learncurve]``)"""
        with pytest.raises(ValueError):
            vak.config.Config.from_toml_path(invalid_train_and_learncurve_config_toml)
