import pathlib

import pytest

import vak.datasets


@pytest.mark.parametrize(
        'dataset_config, split',
        [
            (
                {
                    'name': 'BioSoundSegBench',
                    'params': {
                        'window_size': 2000,
                        'target_type': 'multi_frame_labels',
                        },
                },
                'train',
            ),
            (
                {
                    'name': 'BioSoundSegBench',
                    'params': {
                        'window_size': 2000,
                        'target_type': 'binary_frame_labels',
                        },
                },
                'train',
            ),
            (
                {
                    'name': 'BioSoundSegBench',
                    'params': {
                        'window_size': 2000,
                        'target_type': 'boundary_frame_labels',
                        },
                },
                'train',
            ),
            (
                {
                    'name': 'BioSoundSegBench',
                    'params': {
                        'window_size': 2000,
                        'target_type': 'boundary_frame_labels',
                        },
                },
                'train',
            ),
            (
                {
                    'name': 'BioSoundSegBench',
                    'params': {
                        'window_size': 2000,
                        'target_type': 'boundary_frame_labels',
                        },
                },
                'val',
            ),
            (
                {
                    'name': 'BioSoundSegBench',
                    'params': {
                        'window_size': 2000,
                        'target_type': 'boundary_frame_labels',
                        },
                },
                'test',
            ),

        ]
)
def test_get_biosoundsegbench(dataset_config, split, mock_biosoundsegbench_dataset):
    dataset_path, splits_path = mock_biosoundsegbench_dataset
    dataset_config["path"] = dataset_path
    dataset_config["splits_path"] = splits_path

    dataset = vak.datasets.get(dataset_config, split)

    assert isinstance(dataset, vak.datasets.BioSoundSegBench)
    assert dataset.dataset_path == pathlib.Path(dataset_config["path"])
    assert dataset.splits_path == pathlib.Path(dataset_config["splits_path"])
    assert dataset.window_size == dataset_config["params"]["window_size"]


