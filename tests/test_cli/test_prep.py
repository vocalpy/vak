"""tests for vak.cli.prep module"""
from unittest import mock

import pandas as pd
import pytest

import vak.config
import vak.common.constants
import vak.common.paths


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
        # need to remove dataset_path option from configs we already ran prep on to avoid error
        {
            "section": config_type.upper(),
            "option": "dataset_path",
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

    with mock.patch('vak.prep.prep', autospec=True) as mock_core_prep:
        mock_core_prep.return_value = (pd.DataFrame(), dummy_tmpfile_csv.name)
        vak.cli.prep.prep(toml_path)
        assert mock_core_prep.called


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
def test_prep_dataset_path_raises(
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
