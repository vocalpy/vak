"""tests for vak.cli.learncurve module"""
import pytest

import vak.config
import vak.constants
import vak.cli.learncurve

from . import cli_asserts
from ..test_core.test_learncurve import learncurve_output_matches_expected


def test_learncurve(specific_config, tmp_path, model, device):
    root_results_dir = tmp_path.joinpath("test_learncurve_root_results_dir")
    root_results_dir.mkdir()

    options_to_change = [
        {
            "section": "LEARNCURVE",
            "option": "root_results_dir",
            "value": str(root_results_dir),
        },
        {"section": "LEARNCURVE", "option": "device", "value": device},
    ]

    toml_path = specific_config(
        config_type="learncurve",
        model=model,
        audio_format="cbin",
        annot_format="notmat",
        options_to_change=options_to_change,
    )

    vak.cli.learncurve.learning_curve(toml_path)

    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config_map = vak.config.models.map_from_path(toml_path, cfg.learncurve.models)
    results_path = sorted(root_results_dir.glob(f"{vak.constants.RESULTS_DIR_PREFIX}*"))
    assert len(results_path) == 1
    results_path = results_path[0]

    assert learncurve_output_matches_expected(cfg, model_config_map, results_path)

    assert cli_asserts.toml_config_file_copied_to_results_path(results_path, toml_path)
    assert cli_asserts.log_file_created(command="learncurve", output_path=results_path)


@pytest.mark.parametrize("window_size", [44, 88, 176])
def test_learncurve_previous_run_path(
    specific_config, tmp_path, model, device, previous_run_path_factory, window_size
):
    root_results_dir = tmp_path.joinpath("test_learncurve_root_results_dir")
    root_results_dir.mkdir()

    options_to_change = [
        {
            "section": "LEARNCURVE",
            "option": "root_results_dir",
            "value": str(root_results_dir),
        },
        {"section": "LEARNCURVE", "option": "device", "value": device},
        {
            "section": "LEARNCURVE",
            "option": "previous_run_path",
            "value": str(previous_run_path_factory(model)),
        },
        {"section": "DATALOADER", "option": "window_size", "value": window_size},
    ]

    toml_path = specific_config(
        config_type="learncurve",
        model=model,
        audio_format="cbin",
        annot_format="notmat",
        options_to_change=options_to_change,
    )

    vak.cli.learncurve.learning_curve(toml_path)

    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config_map = vak.config.models.map_from_path(toml_path, cfg.learncurve.models)
    results_path = sorted(root_results_dir.glob(f"{vak.constants.RESULTS_DIR_PREFIX}*"))
    assert len(results_path) == 1
    results_path = results_path[0]

    assert learncurve_output_matches_expected(cfg, model_config_map, results_path)

    assert cli_asserts.toml_config_file_copied_to_results_path(results_path, toml_path)
    assert cli_asserts.log_file_created(command="learncurve", output_path=results_path)
