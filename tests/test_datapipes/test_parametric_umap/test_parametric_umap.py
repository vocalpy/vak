import pytest

import vak
import vak.datapipes.parametric_umap


class TestDatapipe:
    @pytest.mark.parametrize(
        'config_type, model_name, audio_format, spect_format, annot_format, split, transform_kwargs',
        [
            ('train', 'ConvEncoderUMAP', 'cbin', None, 'notmat', 'train', {}),
        ]
    )
    def test_from_dataset_path(self, config_type, model_name, audio_format, spect_format, annot_format,
                               split, transform_kwargs, specific_config_toml_path):
        """Test we can get a :class:`vak.datapipes.parametric_umap.Datapipe` instance
        from the classmethod ``from_dataset_path``"""
        toml_path = specific_config_toml_path(config_type,
                                              model_name,
                                              audio_format=audio_format,
                                              spect_format=spect_format,
                                              annot_format=annot_format)
        cfg = vak.config.Config.from_toml_path(toml_path)
        cfg_command = getattr(cfg, config_type)

        dataset = vak.datapipes.parametric_umap.Datapipe.from_dataset_path(
            dataset_path=cfg_command.dataset.path,
            split=split,
        )
        assert isinstance(dataset, vak.datapipes.parametric_umap.Datapipe)
