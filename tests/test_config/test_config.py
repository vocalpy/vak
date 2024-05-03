import pytest

import vak.config


class TestConfig:
    @pytest.mark.parametrize(
            'tables_to_parse',
            [
                None,
                'prep',
                ['prep'],
            ]
    )
    def test_init_with_real_config(
        self, a_generated_config_dict, tables_to_parse
    ):
        """Test that instantiating Config class works as expected"""
        # this is basically the body of the ``Config.from_config_dict`` function.
        config_kwargs = {}

        if tables_to_parse is None:
            for table_name in a_generated_config_dict:
                config_kwargs[table_name] = vak.config.config.TABLE_CLASSES_MAP[table_name].from_config_dict(
                    a_generated_config_dict[table_name]
                )
        else:
            for table_name in a_generated_config_dict:
                if table_name in tables_to_parse:
                    config_kwargs[table_name] = vak.config.config.TABLE_CLASSES_MAP[table_name].from_config_dict(
                        a_generated_config_dict[table_name]
                    )

        config = vak.config.Config(**config_kwargs)

        assert isinstance(config, vak.config.Config)
        # we already test that config loading works for EvalConfig, et al.,
        # so here we just test that the logic of Config works as expected:
        # we should get an attribute for each top-level table that we pass in;
        # if we don't pass one in, then its corresponding attribute should be None
        for attr in ('eval', 'learncurve', 'predict', 'prep', 'train'):
            if tables_to_parse is not None:
                if attr in a_generated_config_dict and attr in tables_to_parse:
                    assert hasattr(config, attr)
                else:
                    assert getattr(config, attr) is None
            else:
                if attr in a_generated_config_dict:
                    assert hasattr(config, attr)

    @pytest.mark.parametrize(
            'tables_to_parse',
            [
                None,
                'prep',
                ['prep'],
            ]
    )
    def test_from_config_dict_with_real_config(
        self, a_generated_config_dict, tables_to_parse
    ):
        """Test :meth:`Config.from_config_dict`"""
        config = vak.config.Config.from_config_dict(
            a_generated_config_dict, tables_to_parse=tables_to_parse
            )

        assert isinstance(config, vak.config.Config)
        # we already test that config loading works for EvalConfig, et al.,
        # so here we just test that the logic of Config works as expected:
        # we should get an attribute for each top-level table that we pass in;
        # if we don't pass one in, then its corresponding attribute should be None
        for attr in ('eval', 'learncurve', 'predict', 'prep', 'train'):
            if tables_to_parse is not None:
                if attr in a_generated_config_dict and attr in tables_to_parse:
                    assert hasattr(config, attr)
                else:
                    assert getattr(config, attr) is None
            else:
                if attr in a_generated_config_dict:
                    assert hasattr(config, attr)

    @pytest.mark.parametrize(
            'tables_to_parse',
            [
                None,
                'prep',
                ['prep'],
            ]
    )
    def test_from_toml_path(self, a_generated_config_path, tables_to_parse):
        config = vak.config.Config.from_toml_path(
            a_generated_config_path, tables_to_parse=tables_to_parse
            )

        assert isinstance(config, vak.config.Config)

        a_generated_config_dict = vak.config.load._load_toml_from_path(a_generated_config_path)
        # we already test that config loading works for EvalConfig, et al.,
        # so here we just test that the logic of Config works as expected:
        # we should get an attribute for each top-level table that we pass in;
        # if we don't pass one in, then its corresponding attribute should be None
        for attr in ('eval', 'learncurve', 'predict', 'prep', 'train'):
            if tables_to_parse is not None:
                if attr in a_generated_config_dict and attr in tables_to_parse:
                    assert hasattr(config, attr)
                else:
                    assert getattr(config, attr) is None
            else:
                if attr in a_generated_config_dict:
                    assert hasattr(config, attr)

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
