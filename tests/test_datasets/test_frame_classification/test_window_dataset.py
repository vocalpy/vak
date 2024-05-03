import pytest

import vak
import vak.datasets.frame_classification


class TestWindowDataset:
    @pytest.mark.parametrize(
        'config_type, model_name, audio_format, spect_format, annot_format, split, transform_kwargs',
        [
            ('train', 'TweetyNet', 'cbin', None, 'notmat', 'train', {}),
            ('train', 'TweetyNet', None, 'mat', 'yarden', 'train', {}),
        ]
    )
    def test_from_dataset_path(self, config_type, model_name, audio_format, spect_format, annot_format,
                               split, transform_kwargs, specific_config_toml_path):
        """Test we can get a WindowDataset instance from the classmethod ``from_dataset_path``"""
        toml_path = specific_config_toml_path(config_type,
                                              model_name,
                                              audio_format=audio_format,
                                              spect_format=spect_format,
                                              annot_format=annot_format)
        cfg = vak.config.Config.from_toml_path(toml_path)
        cfg_command = getattr(cfg, config_type)

        transform, target_transform = vak.transforms.defaults.get_default_transform(
            model_name, config_type, transform_kwargs
        )

        dataset = vak.datasets.frame_classification.WindowDataset.from_dataset_path(
            dataset_path=cfg_command.dataset.path,
            split=split,
            window_size=cfg_command.train_dataset_params['window_size'],
            transform=transform,
            target_transform=target_transform,
        )
        assert isinstance(dataset, vak.datasets.frame_classification.WindowDataset)
