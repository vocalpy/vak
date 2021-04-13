"""tests for vak.config.learncurve module"""
import vak.config.learncurve
import vak.split


def test_learncurve_attrs_class(all_generated_learncurve_configs_toml):
    """test that instantiating LearncurveConfig class works as expected"""
    for config_toml in all_generated_learncurve_configs_toml:
        learncurve_section = config_toml["LEARNCURVE"]
        config = vak.config.learncurve.LearncurveConfig(**learncurve_section)
        assert isinstance(config, vak.config.learncurve.LearncurveConfig)
