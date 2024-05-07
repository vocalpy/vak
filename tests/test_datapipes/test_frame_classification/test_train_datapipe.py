import pytest

import vak
import vak.datapipes.frame_classification


class TestTrainDatapipe:
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

        dataset = vak.datapipes.frame_classification.TrainDatapipe.from_dataset_path(
            dataset_path=cfg_command.dataset.path,
            split=split,
            window_size=cfg_command.dataset.params['window_size'],
        )
        assert isinstance(dataset, vak.datapipes.frame_classification.TrainDatapipe)
