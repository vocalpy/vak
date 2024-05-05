"""tests for vak.cli.predict module"""
from unittest import mock
import pytest

import vak.cli.predict
import vak.config
import vak.common.constants
import vak.common.paths

from . import cli_asserts


@pytest.mark.parametrize(
    "model_name, audio_format, spect_format, annot_format",
    [
        ("TweetyNet", "cbin", None, "notmat"),
    ],
)
def test_predict(
    model_name, audio_format, spect_format, annot_format, specific_config_toml_path, tmp_path, trainer_table
):
    output_dir = tmp_path.joinpath(
        f"test_predict_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    keys_to_change = [
        {"table": "predict", "key": "output_dir", "value": str(output_dir)},
        {"table": "predict", "key": "trainer", "value": trainer_table},
    ]

    toml_path = specific_config_toml_path(
        config_type="predict",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        keys_to_change=keys_to_change,
    )

    with mock.patch('vak.predict.predict', autospec=True) as mock_core_predict:
        vak.cli.predict.predict(toml_path)
        assert mock_core_predict.called

    assert cli_asserts.log_file_created(command="predict", output_path=output_dir)
    assert cli_asserts.log_file_contains_version(command="predict", output_path=output_dir)


def test_predict_dataset_none_raises(
        specific_config_toml_path
):
    """Test that cli.predict raises ValueError when dataset_path is None
    (presumably because `vak prep` was not run yet)
    """
    keys_to_change = [
        {"table": "predict", "key": "dataset", "value": "DELETE-KEY"},
    ]

    toml_path = specific_config_toml_path(
        config_type="predict",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        keys_to_change=keys_to_change,
    )

    with pytest.raises(KeyError):
        vak.cli.predict.predict(toml_path)
