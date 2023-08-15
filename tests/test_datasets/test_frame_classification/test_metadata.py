import json
import pathlib

import pytest

import vak.datasets.frame_classification


ARGNAMES = 'dataset_csv_filename, timebin_dur'
ARGVALS = [
    (pathlib.Path('bird1_prep_230319_115852.csv'), 0.002),
    (pathlib.Path('bird1_prep_230319_115852.csv'), 0.001),
    (pathlib.Path('bird1_prep_230319_115852.csv'), 0.0027),
]


class TestMetadata:
    @pytest.mark.parametrize(
        ARGNAMES,
        ARGVALS
    )
    def test_metadata_init(self, dataset_csv_filename, timebin_dur):
        metadata = vak.datasets.frame_classification.Metadata(dataset_csv_filename, timebin_dur)
        assert isinstance(metadata, vak.datasets.frame_classification.Metadata)
        for attr_name, attr_val in zip(
            ('dataset_csv_filename', 'timebin_dur'),
            (dataset_csv_filename, timebin_dur),
        ):
            assert hasattr(metadata, attr_name)
            if isinstance(attr_val, pathlib.Path):
                assert getattr(metadata, attr_name) == str(attr_val)
            else:
                assert getattr(metadata, attr_name) == attr_val

    @pytest.mark.parametrize(
        ARGNAMES,
        ARGVALS
    )
    def test_metadata_from_path(self, dataset_csv_filename, timebin_dur, tmp_path):
        # we make metadata "by hand"
        metadata_dict = {
            'dataset_csv_filename': str(dataset_csv_filename),
            'timebin_dur': timebin_dur,
        }
        metadata_json_path = tmp_path / vak.datasets.frame_classification.Metadata.METADATA_JSON_FILENAME
        with metadata_json_path.open('w') as fp:
            json.dump(metadata_dict, fp, indent=4)

    @pytest.mark.parametrize(
        ARGNAMES,
        ARGVALS
    )
    def test_metadata_to_json(self, dataset_csv_filename, timebin_dur, tmp_path):
        metadata_to_json = vak.datasets.frame_classification.Metadata(dataset_csv_filename, timebin_dur)
        mock_dataset_path = tmp_path / 'mock_dataset'
        mock_dataset_path.mkdir()

        metadata_to_json.to_json(dataset_path=mock_dataset_path)
        expected_json_path = mock_dataset_path / vak.datasets.frame_classification.Metadata.METADATA_JSON_FILENAME
        assert expected_json_path.exists()

        metadata_from_json = vak.datasets.frame_classification.Metadata.from_path(expected_json_path)
        assert metadata_from_json == metadata_to_json
