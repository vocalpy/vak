import pytest

import vak
import vak.datasets


class TestWindowDataset:
    @pytest.mark.parametrize(
        'config_type, model_name, audio_format, spect_format, annot_format, x_source',
        [
            ('train', 'teenytweetynet', 'cbin', None, 'notmat', 'spect'),
            ('train', 'teenytweetynet', None, 'mat', 'yarden', 'spect'),
            ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 'spect'),
        ]
    )
    def test_from_csv(self, config_type, model_name, audio_format, spect_format, annot_format, x_source,
                      specific_config):
        """Test we can get a WindowDataset instance from the classmethod ``from_csv``

        This is the way we make ``WindowDataset`` instances
        inside ``vak.core.train``,
        as opposed to when we *also* pass in vectors representing the windows,
        which we do in ``vak.core.learncurve.learning_curve``,
        see next test method.
        """
        toml_path = specific_config(config_type,
                                    model_name,
                                    audio_format=audio_format,
                                    spect_format=spect_format,
                                    annot_format=annot_format)
        cfg = vak.config.parse.from_toml_path(toml_path)
        cfg_command = getattr(cfg, config_type)

        # stuff we need just to be able to instantiate window dataset
        labelmap = vak.labels.to_map(cfg.prep.labelset, map_unlabeled=True)

        transform, target_transform = vak.transforms.get_defaults('train')

        metadata = vak.datasets.metadata.Metadata.from_dataset_path(cfg_command.dataset_path)
        dataset_csv_path = cfg_command.dataset_path / metadata.dataset_csv_filename

        dataset = vak.datasets.WindowDataset.from_csv(
            dataset_csv_path=dataset_csv_path,
            split='train',
            labelmap=labelmap,
            window_size=cfg.dataloader.window_size,
            transform=transform,
            target_transform=target_transform,
            source_ids=None,
            source_inds=None,
            window_inds=None,
        )
        assert isinstance(dataset, vak.datasets.WindowDataset)

    def test_from_csv_with_vectors(self, window_dataset_from_csv_kwargs):
        """Test that classmethod ``WindowDataset.from_csv`` works
        when we pass in vectors representing windows.

        This is the way we make ``WindowDataset`` instances
        inside ``vak.core.learncurve.learning_curve``.

        We get the vectors "by hand" inside the ``learning_curve``
        function, and then feed them in to the ``from_csv``
        classmethod when we instantiate.
        """
        transform, target_transform = vak.transforms.get_defaults('train')
        dataset = vak.datasets.WindowDataset.from_csv(
            split='train',
            transform=transform,
            target_transform=target_transform,
            **window_dataset_from_csv_kwargs
        )
        assert isinstance(dataset, vak.datasets.WindowDataset)
