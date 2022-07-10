import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def specific_prep_csv_path(specific_config_toml):
    """returns a function that will return the
    ``csv_path`` corresponding to the ``prep``ared dataset
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
        return Path(config_toml[config_type.upper()]["csv_path"])
        _return_toml(config_path)

    return _specific_csv_path


@pytest.fixture
def dummy_tmpfile_csv():
    with tempfile.NamedTemporaryFile() as fp:
        yield fp
