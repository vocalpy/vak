"""tests for vak.cli.predict module"""
from unittest import mock
import pytest

import vak.cli.predict
import vak.config
import vak.constants
import vak.paths

from . import cli_asserts


@pytest.mark.parametrize(
    "audio_format, spect_format, annot_format",
    [
        ("cbin", None, "notmat"),
        ("wav", None, "birdsong-recognition-dataset"),
    ],
)
def test_predict(
    audio_format, spect_format, annot_format, specific_config, tmp_path, model, device
):
    output_dir = tmp_path.joinpath(
        f"test_predict_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "PREDICT", "option": "output_dir", "value": str(output_dir)},
        {"section": "PREDICT", "option": "device", "value": device},
    ]

    toml_path = specific_config(
        config_type="predict",
        model=model,
        audio_format=audio_format,
        annot_format=annot_format,
        options_to_change=options_to_change,
    )

    with mock.patch('vak.core.predict', autospec=True) as mock_core_predict:
        vak.cli.predict.predict(toml_path)
        assert mock_core_predict.called

    assert cli_asserts.log_file_created(command="predict", output_path=output_dir)
    assert cli_asserts.log_file_contains_version(command="predict", output_path=output_dir)


def test_predict_csv_path_none_raises(
        specific_config, tmp_path,
):
    """Test that cli.predict raises ValueError when csv_path is None
    (presumably because `vak prep` was not run yet)
    """
    options_to_change = [
        {"section": "PREDICT", "option": "csv_path", "value": "DELETE-OPTION"},
    ]

    toml_path = specific_config(
        config_type="predict",
        model="teenytweetynet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=options_to_change,
    )

    with pytest.raises(ValueError):
        vak.cli.predict.predict(toml_path)
