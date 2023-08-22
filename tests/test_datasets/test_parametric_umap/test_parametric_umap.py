import pytest

import vak
import vak.datasets.parametric_umap


class TestParametricUMAPDataset:
    @pytest.mark.parametrize(
        'config_type, model_name, audio_format, spect_format, annot_format, split, transform_kwargs',
        [
            ('train', 'ConvEncoderUMAP', 'cbin', None, 'notmat', 'train', {}),
        ]
    )
    def test_from_dataset_path(self, config_type, model_name, audio_format, spect_format, annot_format,
                               split, transform_kwargs, specific_config):
        """Test we can get a WindowDataset instance from the classmethod ``from_dataset_path``"""
        toml_path = specific_config(config_type,
                                    model_name,
                                    audio_format=audio_format,
                                    spect_format=spect_format,
                                    annot_format=annot_format)
        cfg = vak.config.parse.from_toml_path(toml_path)
        cfg_command = getattr(cfg, config_type)

        transform = vak.transforms.defaults.get_default_transform(
            model_name, config_type, transform_kwargs
        )

        dataset = vak.datasets.parametric_umap.ParametricUMAPDataset.from_dataset_path(
            dataset_path=cfg_command.dataset_path,
            split=split,
            transform=transform,
        )
        assert isinstance(dataset, vak.datasets.parametric_umap.ParametricUMAPDataset)
