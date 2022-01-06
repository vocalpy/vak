"""tests for vak.cli.train module"""
import pytest

import vak.config
import vak.constants
import vak.paths
import vak.cli.train

from . import cli_asserts
from ..test_core.test_train import train_output_matches_expected


@pytest.mark.parametrize(
    "audio_format, spect_format, annot_format",
    [
        ("cbin", None, "notmat"),
        ("wav", None, "birdsong-recognition-dataset"),
        (None, "mat", "yarden"),
    ],
)
def test_train(
    audio_format, spect_format, annot_format, specific_config, tmp_path, model, device
):
    root_results_dir = tmp_path.joinpath("test_train_root_results_dir")
    root_results_dir.mkdir()

    options_to_change = [
        {
            "section": "TRAIN",
            "option": "root_results_dir",
            "value": str(root_results_dir),
        },
        {"section": "TRAIN", "option": "device", "value": device},
    ]

    toml_path = specific_config(
        config_type="train",
        model=model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )

    vak.cli.train.train(toml_path)

    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config_map = vak.config.models.map_from_path(toml_path, cfg.train.models)
    results_path = sorted(root_results_dir.glob(f"{vak.constants.RESULTS_DIR_PREFIX}*"))
    assert len(results_path) == 1
    results_path = results_path[0]

    assert train_output_matches_expected(cfg, model_config_map, results_path)

    assert cli_asserts.toml_config_file_copied_to_results_path(results_path, toml_path)
    assert cli_asserts.log_file_created(command="train", output_path=results_path)
