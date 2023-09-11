import pytest

import vak
import vak.datasets.frame_classification


class TestWindowDataset:
    @pytest.mark.parametrize(
        'config_type, model_name, audio_format, spect_format, annot_format, split',
        [
            ('eval', 'TweetyNet', 'cbin', None, 'notmat', 'test'),
        ]
    )
    def test_from_dataset_path(self, config_type, model_name, audio_format, spect_format, annot_format,
                               split, specific_config):
        """Test we can get a FramesDataset instance from the classmethod ``from_dataset_path``"""
        toml_path = specific_config(config_type,
                                    model_name,
                                    audio_format=audio_format,
                                    spect_format=spect_format,
                                    annot_format=annot_format)
        cfg = vak.config.parse.from_toml_path(toml_path)
        cfg_command = getattr(cfg, config_type)

        item_transform = vak.transforms.defaults.get_default_transform(
            model_name, config_type, cfg.eval.transform_params
        )

        dataset = vak.datasets.frame_classification.FramesDataset.from_dataset_path(
            dataset_path=cfg_command.dataset_path,
            split=split,
            item_transform=item_transform,
        )
        assert isinstance(dataset, vak.datasets.frame_classification.FramesDataset)
