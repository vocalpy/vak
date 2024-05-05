import pytest

import vak.config.trainer


class TestTrainerConfig:

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'accelerator': 'cpu',
                },
                {
                    'accelerator': 'gpu',
                    'devices': [0],
                },
                {
                    'accelerator': 'gpu',
                    'devices': [1],
                },
                {
                    'accelerator': 'cpu',
                    'devices': 1,
                },
                {
                    'devices': 1,
                },
            ]
    )
    def test_init(self, config_dict):
        trainer_config = vak.config.trainer.TrainerConfig(**config_dict)

        assert isinstance(trainer_config, vak.config.trainer.TrainerConfig)
        if 'accelerator' in config_dict:
            assert getattr(trainer_config, 'accelerator') == config_dict['accelerator']
        else:
            # TODO: mock `accelerator.get_default` here, return either 'cpu' or 'gpu'
            assert getattr(trainer_config, 'accelerator') == vak.common.accelerator.get_default()
        if 'devices' in config_dict:
            assert getattr(trainer_config, 'devices') == config_dict['devices']
        else:
            if 'accelerator' == 'cpu':
                assert getattr(trainer_config, 'devices') == 1
            elif 'accelerator' in ('gpu', 'tpu', 'ipu'):
                assert getattr(trainer_config, 'devices') == [0]

    @pytest.mark.parametrize(
            'config_dict, expected_exception',
            [
                # 'accelerator' can't be 'auto', breaks train/eval/prep/learncurve functions
                (
                    {
                        'accelerator': 'auto',
                    },
                    ValueError,
                ),
                # throws a device because parallel across GPUs won't work right now
                (
                    {
                        'accelerator': 'gpu',
                        'devices': [0, 1],
                    },
                    ValueError,
                ),
                # 'devices' can't be -1, breaks train/eval/prep/learncurve functions
                (
                    {
                        'accelerator': 'gpu',
                        'devices': -1,
                    },
                    ValueError,
                ),
                (
                    {
                        'accelerator': 'gpu',
                        'devices': 'auto',
                    },
                    ValueError,
                ),
                # when accelerator is CPU, devices must be int gt 0
                (
                    {
                        'accelerator': 'cpu',
                        'devices': -1,
                    },
                    ValueError,
                )
            ]
    )
    def test_init_raises(self, config_dict, expected_exception):
        with pytest.raises(expected_exception):
            vak.config.trainer.TrainerConfig(**config_dict)

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'accelerator': 'cpu',
                },
                {
                    'accelerator': 'gpu',
                },
                {
                    'accelerator': 'gpu',
                    'devices': [0],
                },
                {
                    'accelerator': 'gpu',
                    'devices': [1],
                },
                {
                    'accelerator': 'cpu',
                    'devices': 1,
                },
                {
                    'devices': 1,
                },
            ]
    )
    def test_asdict(self, config_dict):
        trainer_config = vak.config.trainer.TrainerConfig(**config_dict)

        trainer_config_asdict = trainer_config.asdict()

        assert isinstance(trainer_config_asdict, dict)
        if 'accelerator' in config_dict:
            assert trainer_config_asdict['accelerator'] == config_dict['accelerator']
        else:
            # TODO: mock `accelerator.get_default` here, return either 'cpu' or 'gpu'
            assert trainer_config_asdict['accelerator'] == vak.common.accelerator.get_default()
        if 'devices' in config_dict:
            assert trainer_config_asdict['devices'] == config_dict['devices']
        else:
            if config_dict["accelerator"] == 'cpu':
                assert trainer_config_asdict['devices'] == 1
            elif config_dict["accelerator"] == 'gpu':
                assert trainer_config_asdict['devices'] == [0]
