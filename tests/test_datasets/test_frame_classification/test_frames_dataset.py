import pytest

import vak
import vak.datapipes.frame_classification


class TestWindowDataset:
    @pytest.mark.parametrize(
        'config_type, model_name, audio_format, spect_format, annot_format, split',
        [
            ('eval', 'TweetyNet', 'cbin', None, 'notmat', 'test'),
        ]
    )
    def test_from_dataset_path(self, config_type, model_name, audio_format, spect_format, annot_format,
                               split, specific_config_toml_path):
        """Test we can get a FramesDataset instance from the classmethod ``from_dataset_path``"""
        toml_path = specific_config_toml_path(config_type,
                                              model_name,
                                              audio_format=audio_format,
                                              spect_format=spect_format,
                                              annot_format=annot_format)
        cfg = vak.config.Config.from_toml_path(toml_path)
        cfg_command = getattr(cfg, config_type)

        transform_kwargs = {
            "window_size": cfg.eval.dataset.params["window_size"]
        }
        item_transform = vak.transforms.defaults.get_default_transform(
            model_name, config_type, transform_kwargs
        )

        dataset = vak.datapipes.frame_classification.FramesDataset.from_dataset_path(
            dataset_path=cfg_command.dataset.path,
            split=split,
            item_transform=item_transform,
        )
        assert isinstance(dataset, vak.datapipes.frame_classification.FramesDataset)
