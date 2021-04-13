"""tests for vak.config.predict module"""
import vak.config.predict
import vak.split


def test_predict_attrs_class(all_generated_predict_configs_toml):
    """test that instantiating PredictConfig class works as expected"""
    for config_toml in all_generated_predict_configs_toml:
        predict_section = config_toml["PREDICT"]
        config = vak.config.predict.PredictConfig(**predict_section)
        assert isinstance(config, vak.config.predict.PredictConfig)
