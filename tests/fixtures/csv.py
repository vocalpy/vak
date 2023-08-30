import tempfile
from pathlib import Path

import pytest

import vak


@pytest.fixture
def specific_dataset_csv_path(specific_config_toml):
    """Returns a function that will return the
    ``dataset_csv_path`` corresponding to the ``prep``ared dataset
    from a specific configuration file,
    determined by characteristics specified by the caller:
    `config_type`, `audio_format`, `spect_format`, `annot_format`
    """

    def _specific_csv_path(
        config_type,
        model,
        annot_format,
        audio_format=None,
        spect_format=None,
    ):
        config_toml = specific_config_toml(
            config_type, model, annot_format, audio_format, spect_format
        )
        dataset_path = Path(config_toml[config_type.upper()]["dataset_path"])
        # TODO: make this more general -- dataset registry?
        if config_toml['PREP']['dataset_type'] == 'frame classification':
            metadata = vak.datasets.frame_classification.Metadata.from_dataset_path(dataset_path)
        elif config_toml['PREP']['dataset_type'] == 'parametric umap':
            metadata = vak.datasets.parametric_umap.Metadata.from_dataset_path(dataset_path)
        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        return dataset_csv_path

    return _specific_csv_path


@pytest.fixture
def dummy_tmpfile_csv():
    with tempfile.NamedTemporaryFile() as fp:
        yield fp
