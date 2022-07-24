"""tests for vak.config.eval module"""
import vak.config.eval


def test_predict_attrs_class(all_generated_eval_configs_toml):
    """test that instantiating EvalConfig class works as expected"""
    for config_toml in all_generated_eval_configs_toml:
        eval_section = config_toml["EVAL"]
        config = vak.config.eval.EvalConfig(**eval_section)
        assert isinstance(config, vak.config.eval.EvalConfig)
