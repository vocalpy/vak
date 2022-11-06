"""tests for vak.cli.prep module"""
from unittest import mock

from pathlib import Path

import pandas as pd
import pytest

import vak.config
import vak.constants
import vak.core.train
import vak.paths
import vak.io.spect

from . import cli_asserts


@pytest.mark.parametrize(
    "config_type, audio_format, spect_format, annot_format",
    [
        ("eval", "cbin", None, "notmat"),
        ("learncurve", "cbin", None, "notmat"),
        ("predict", "cbin", None, "notmat"),
        ("predict", "wav", None, "birdsong-recognition-dataset"),
        ("train", "cbin", None, "notmat"),
        ("train", "wav", None, "birdsong-recognition-dataset"),
        ("train", None, "mat", "yarden"),
    ],
)
def test_purpose_from_toml(
    config_type,
    audio_format,
    spect_format,
    annot_format,
    specific_config,
    default_model,
    tmp_path,
):
    toml_path = specific_config(
        config_type=config_type,
        model=default_model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
    )
    config_toml = vak.config.parse._load_toml_from_path(toml_path)
    vak.cli.prep.purpose_from_toml(config_toml)


@pytest.mark.parametrize(
    "config_type, audio_format, spect_format, annot_format",
    [
        ("eval", "cbin", None, "notmat"),
        ("learncurve", "cbin", None, "notmat"),
        ("predict", "cbin", None, "notmat"),
        ("predict", "wav", None, "birdsong-recognition-dataset"),
        ("train", "cbin", None, "notmat"),
        ("train", "wav", None, "birdsong-recognition-dataset"),
        ("train", None, "mat", "yarden"),
    ],
)
def test_prep(
    config_type,
    audio_format,
    spect_format,
    annot_format,
    specific_config,
    default_model,
    tmp_path,
    dummy_tmpfile_csv,
):
    output_dir = tmp_path.joinpath(
        f"test_prep_{config_type}_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "PREP", "option": "output_dir", "value": str(output_dir)},
        # need to remove csv_path option from configs we already ran prep on to avoid error
        {
            "section": config_type.upper(),
            "option": "csv_path",
            "value": None,
        },
    ]
    toml_path = specific_config(
        config_type=config_type,
        model=default_model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )

    with mock.patch('vak.core.prep', autospec=True) as mock_core_prep:
        mock_core_prep.return_value = (pd.DataFrame(), dummy_tmpfile_csv.name)
        vak.cli.prep.prep(toml_path)
        assert mock_core_prep.called

    cfg = vak.config.parse.from_toml_path(toml_path)
    command_section = getattr(cfg, config_type)
    csv_path = getattr(command_section, "csv_path")
    # we don't bother checking whether csv is as expected
    # because that's already tested by `test_io.test_spect`, `test_io.test_dataframe`, etc.
    assert Path(csv_path).exists()

    assert cli_asserts.log_file_created(command="prep", output_path=cfg.prep.output_dir)
    assert cli_asserts.log_file_contains_version(command="prep", output_path=output_dir)


@pytest.mark.parametrize(
    "config_type, audio_format, spect_format, annot_format",
    [
        ("eval", "cbin", None, "notmat"),
        ("learncurve", "cbin", None, "notmat"),
        ("predict", "cbin", None, "notmat"),
        ("predict", "wav", None, "birdsong-recognition-dataset"),
        ("train", "cbin", None, "notmat"),
        ("train", "wav", None, "birdsong-recognition-dataset"),
        ("train", None, "mat", "yarden"),
    ],
)
def test_prep_csv_path_raises(
    config_type,
    audio_format,
    spect_format,
    annot_format,
    specific_config,
    default_model,
    tmp_path,

):
    output_dir = tmp_path.joinpath(
        f"test_prep_{config_type}_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "PREP", "option": "output_dir", "value": str(output_dir)},
    ]
    toml_path = specific_config(
        config_type=config_type,
        model=default_model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )

    with pytest.raises(ValueError):
        vak.cli.prep.prep(toml_path)
