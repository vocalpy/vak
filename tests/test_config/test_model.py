import pytest

import vak.config.model


class TestModelConfig:

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'NonExistentModel': {
                        'network': {},
                        'optimizer': {},
                        'loss': {},
                        'metrics': {},
                    }
                },
                {
                    'TweetyNet': {
                        'network': {},
                        'optimizer': {'lr': 1e-3},
                        'loss': {},
                        'metrics': {},
                    }
                },
            ]
    )
    def test_init(self, config_dict):
        name=list(config_dict.keys())[0]
        config_dict_from_name = config_dict[name]

        model_config = vak.config.model.ModelConfig(
            name=name,
            **config_dict_from_name
            )

        assert isinstance(model_config, vak.config.model.ModelConfig)
        assert model_config.name == name
        for key, val in config_dict_from_name.items():
            assert hasattr(model_config, key)
            assert getattr(model_config, key) == val

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'TweetyNet': {
                        'optimizer': {'lr': 1e-3},
                    }
                },
                {
                    'TweetyNet': {
                        'network': {},
                        'optimizer': {'lr': 1e-3},
                        'loss': {},
                        'metrics': {},
                    }
                },
                {
                    'ED_TCN': {
                        'optimizer': {'lr': 1e-3},
                    }
                },
                {
                    "ConvEncoderUMAP": {
                        "optimizer": {'lr': 1e-3},
                    }
                },
                {
                    "ConvEncoderUMAP": {
                        "network": {
                            "encoder": {
                                "conv1_filters": 8,
                            }
                        },
                        "optimizer": {'lr': 1e-3},
                    }
                }
            ]
    )
    def test_from_config_dict(self, config_dict):
        model_config = vak.config.model.ModelConfig.from_config_dict(config_dict)

        name=list(config_dict.keys())[0]
        config_dict_from_name = config_dict[name]
        assert model_config.name == name
        for attr in ('network', 'optimizer', 'loss', 'metrics'):
            assert hasattr(model_config, attr)
            if attr in config_dict_from_name:
                assert getattr(model_config, attr) == config_dict_from_name[attr]
            else:
                assert getattr(model_config, attr) == {}

    def test_from_config_dict_real_config(self, a_generated_config_dict):
        config_dict = None
        for table_name in ('train', 'eval', 'predict', 'learncurve'):
            if table_name in a_generated_config_dict:
                config_dict = a_generated_config_dict[table_name]['model']
        if config_dict is None:
            raise ValueError(
                f"Didn't find top-level table for config: {a_generated_config_dict}"
            )

        model_config = vak.config.model.ModelConfig.from_config_dict(config_dict)

        name=list(config_dict.keys())[0]
        config_dict_from_name = config_dict[name]
        assert model_config.name == name
        for attr in ('network', 'optimizer', 'loss', 'metrics'):
            assert hasattr(model_config, attr)
            if attr in config_dict_from_name:
                assert getattr(model_config, attr) == config_dict_from_name[attr]
            else:
                assert getattr(model_config, attr) == {}

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'TweetyNet': {
                        'optimizer': {'lr': 1e-3},
                    }
                },
                {
                    'TweetyNet': {
                        'network': {},
                        'optimizer': {'lr': 1e-3},
                        'loss': {},
                        'metrics': {},
                    }
                },
                {
                    'ED_TCN': {
                        'optimizer': {'lr': 1e-3},
                    }
                },
                {
                    "ConvEncoderUMAP": {
                        "optimizer": {'lr': 1e-3},
                    }
                },
                {
                    "ConvEncoderUMAP": {
                        "network": {
                            "encoder": {
                                "conv1_filters": 8,
                            }
                        },
                        "optimizer": {'lr': 1e-3},
                    }
                }
            ]
    )
    def test_asdict(self, config_dict):
        model_config = vak.config.model.ModelConfig.from_config_dict(config_dict)

        model_config_as_dict = model_config.asdict()

        assert isinstance(model_config_as_dict, dict)

        model_name = list(config_dict.keys())[0]
        for key in ('name', 'network', 'optimizer', 'loss', 'metrics'):
            if key == 'name':
                assert model_config_as_dict[key] == model_name
            else:
                if key in config_dict[model_name]:
                    assert model_config_as_dict[key] == config_dict[model_name][key]
                else:
                    assert model_config_as_dict[key] == {}

    @pytest.mark.parametrize(
            'config_dict, expected_exception',
            [
                # Raises ValueError because model name not in registry
                (
                    {
                        'NonExistentModel': {
                            'network': {},
                            'optimizer': {},
                            'loss': {},
                            'metrics': {},
                        }
                    },
                    ValueError,
                )
            ]
    )
    def test_from_config_dict_raises(self, config_dict, expected_exception):
        with pytest.raises(expected_exception):
            vak.config.model.ModelConfig.from_config_dict(config_dict)
