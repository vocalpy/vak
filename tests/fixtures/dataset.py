from pathlib import Path

import pytest


@pytest.fixture
def specific_dataset_path(specific_config_toml):
    """Returns a function that will return the
    ``dataset_path`` corresponding to the ``prep``ared dataset
    from a specific configuration file,
    determined by characteristics specified by the caller:
    `config_type`, `audio_format`, `spect_format`, `annot_format`
    """

    def _specific_dataset_path(
        config_type,
        model,
        annot_format,
        audio_format=None,
        spect_format=None,
    ):
        config_toml = specific_config_toml(
            config_type, model, annot_format, audio_format, spect_format
        )
        dataset_path = Path(config_toml[config_type]["dataset"]["path"])
        return dataset_path

    return _specific_dataset_path
