"""tests for vak.config.train module"""
import vak.config.train


def test_train_attrs_class(all_generated_train_configs_toml_path_pairs):
    """test that instantiating TrainConfig class works as expected"""
    for config_toml, toml_path in all_generated_train_configs_toml_path_pairs:
        train_section = config_toml["TRAIN"]
        train_config_obj = vak.config.train.TrainConfig(**train_section)
        assert isinstance(train_config_obj, vak.config.train.TrainConfig)
