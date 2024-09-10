
import pytest

import vak.datasets


class TestCMACBench:
    @pytest.mark.parametrize(
            'split, window_size, target_type',
            [
                (
                    "train",
                    2000,
                    "multi_frame_labels",
                ),
                (
                    "train",
                    2000,
                    "binary_frame_labels",
                ),
                (
                    "train",
                    2000,
                    "boundary_frame_labels",
                ),
                (
                    "train",
                    2000,
                    ("multi_frame_labels", "boundary_frame_labels"),
                ),
            ]
    )
    def test_init(self, split, window_size, target_type, mock_CMACBench_dataset):
        dataset_path, splits_path = mock_CMACBench_dataset
        dataset = vak.datasets.CMACBench(
            dataset_path,
            splits_path,
            split,
            window_size,
            target_type,
        )
        assert isinstance(dataset, vak.datasets.CMACBench)

    @pytest.mark.parametrize(
            'dataset_path, splits_path, split, window_size, target_type, expected_exception',
            [
                # invalid dataset path -> NotADirectoryError
                (
                    'path/to/dataset/that/doesnt/exist',
                    None,
                    "train",
                    2000,
                    "multi_frame_labels",
                    NotADirectoryError,
                ),
                # invalid splits path -> FileNotFoundError
                (
                    None,
                    'path/to/splits/that/doesnt/exist',
                    "train",
                    2000,
                    "binary_frame_labels",
                    FileNotFoundError,
                ),
                # invalid split -> ValueError
                (
                    None,
                    None,
                    "evaluate",
                    2000,
                    "boundary_frame_labels",
                    ValueError,
                ),
                # no target type when split != "predict" -> ValueError
                (
                    None,
                    None,
                    "train",
                    2000,
                    None,
                    ValueError,
                ),
                # wrong type for target type -> TypeError
                (
                    None,
                    None,
                    "train",
                    2000,
                    1,
                    TypeError,
                ),
                # wrong type for target type -> TypeError
                (
                    None,
                    None,
                    "train",
                    2000,
                    ("boundary_frame_labels", 1),
                    TypeError,
                ),
                # invalid target type -> ValueError
                (
                    None,
                    None,
                    "train",
                    2000,
                    "frame_labels",
                    ValueError,
                ),
            ]
    )
    def test_init_raises(
            self, dataset_path, splits_path, split, window_size, target_type, expected_exception, mock_CMACBench_dataset
        ):
        if dataset_path is None and splits_path is not None:
            dataset_path, _ = mock_CMACBench_dataset
        elif dataset_path is not None and splits_path is None:
            _, splits_path = mock_CMACBench_dataset
        elif dataset_path is None and splits_path is None:
            dataset_path, splits_path = mock_CMACBench_dataset

        with pytest.raises(expected_exception):
            dataset = vak.datasets.CMACBench(
                dataset_path,
                splits_path,
                split,
                window_size,
                target_type,
            )
