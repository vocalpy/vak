"""tests for vak.cli.predict module"""
import pytest

import vak.cli.predict
import vak.config
import vak.constants
import vak.paths

from . import cli_asserts
from ..test_core.test_predict import predict_output_matches_expected


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

    vak.cli.predict.predict(toml_path)

    cfg = vak.config.parse.from_toml_path(toml_path)
    assert predict_output_matches_expected(output_dir, cfg.predict.annot_csv_filename)

    assert cli_asserts.log_file_created(command="predict", output_path=output_dir)
