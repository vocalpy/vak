"""tests for vak.cli.eval module"""
from unittest import mock

import pytest

import vak.cli.eval
import vak.config
import vak.constants
import vak.paths

from . import cli_asserts


@pytest.mark.parametrize(
    "audio_format, spect_format, annot_format",
    [
        ("cbin", None, "notmat"),
    ],
)
def test_eval(
    audio_format, spect_format, annot_format, specific_config, tmp_path, model, device
):
    output_dir = tmp_path.joinpath(
        f"test_eval_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "EVAL", "option": "output_dir", "value": str(output_dir)},
        {"section": "EVAL", "option": "device", "value": device},
    ]

    toml_path = specific_config(
        config_type="eval",
        model=model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )

    cfg = vak.config.parse.from_toml_path(toml_path)

    with mock.patch('vak.core.eval', autospec=True) as mock_core_eval:
        vak.cli.eval.eval(toml_path)
        assert mock_core_eval.called

    assert cli_asserts.log_file_created(command="eval", output_path=output_dir)
