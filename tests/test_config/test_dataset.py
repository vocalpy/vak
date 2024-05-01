import pathlib

import pytest

import vak.config.dataset


class TestDatasetConfig:
    @pytest.mark.parametrize(
        'path, splits_path, name',
        [
            # typical use by a user with default split
            ('~/user/prepped/dataset', None, None),
            # use by a user with a split specified
            ('~/user/prepped/dataset', 'spilts/replicate-1.json', None),
            # use of a built-in dataset, with a split specified
            ('~/datasets/BioSoundSegBench', 'splits/Bengalese-Finch-song-gy6or6-replicate-1.json', 'BioSoundSegBench'),

        ]
    )
    def test_init(self, path, splits_path, name):
        if name is None and splits_path is None:
            dataset_config = vak.config.dataset.DatasetConfig(
                path=path
            )
        elif name is None:
            dataset_config = vak.config.dataset.DatasetConfig(
                path=path,
                splits_path=splits_path,
            )
        else:
            dataset_config = vak.config.dataset.DatasetConfig(
                name=name,
                path=path,
                splits_path=splits_path,
            )
        assert isinstance(dataset_config, vak.config.dataset.DatasetConfig)
        assert dataset_config.path == pathlib.Path(path)
        if splits_path is not None:
            assert dataset_config.splits_path == pathlib.Path(splits_path)
        else:
            assert dataset_config.splits_path is None
        if name is not None:
            assert dataset_config.name == name
        else:
            assert dataset_config.name is None

    @pytest.mark.parametrize(
        'config_dict',
        [
            {
                'path' :'~/datasets/BioSoundSegBench',
                'splits_path': 'splits/Bengalese-Finch-song-gy6or6-replicate-1.json',
                'name': 'BioSoundSegBench',
            },
            {
                'path' :'~/user/prepped/dataset',
            },
            {
                'path' :'~/user/prepped/dataset',
                'splits_path': 'splits/replicate-1.json'
            },
        ]
    )
    def test_from_config_dict(self, config_dict):
        dataset_config = vak.config.dataset.DatasetConfig.from_config_dict(config_dict)
        assert isinstance(dataset_config, vak.config.dataset.DatasetConfig)
        assert dataset_config.path == pathlib.Path(config_dict['path'])
        if 'splits_path' in config_dict:
            assert dataset_config.splits_path == pathlib.Path(config_dict['splits_path'])
        else:
            assert dataset_config.splits_path is None
        if 'name' in config_dict:
            assert dataset_config.name == config_dict['name']
        else:
            assert dataset_config.name is None

