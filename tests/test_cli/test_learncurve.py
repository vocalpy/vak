"""tests for vak.cli.learncurve module"""
from unittest import mock

import pytest

import vak.config
import vak.common.constants
import vak.cli.learncurve

from . import cli_asserts


def test_learncurve(specific_config_toml_path, tmp_path, trainer_table):
    root_results_dir = tmp_path.joinpath("test_learncurve_root_results_dir")
    root_results_dir.mkdir()

    keys_to_change = [
        {
            "table": "learncurve",
            "key": "root_results_dir",
            "value": str(root_results_dir),
        },
        {"table": "learncurve", "key": "trainer", "value": trainer_table},
    ]

    toml_path = specific_config_toml_path(
        config_type="learncurve",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        keys_to_change=keys_to_change,
    )

    with mock.patch('vak.learncurve.learning_curve', autospec=True) as mock_core_learning_curve:
        vak.cli.learncurve.learning_curve(toml_path)
        assert mock_core_learning_curve.called

    results_path = sorted(root_results_dir.glob(f"{vak.common.constants.RESULTS_DIR_PREFIX}*"))
    assert len(results_path) == 1
    results_path = results_path[0]

    assert cli_asserts.toml_config_file_copied_to_results_path(results_path, toml_path)
    assert cli_asserts.log_file_created(command="learncurve", output_path=results_path)
    assert cli_asserts.log_file_contains_version(command="learncurve", output_path=results_path)


def test_learning_curve_dataset_none_raises(
        specific_config_toml_path, tmp_path,
):
    """Test that cli.learncurve.learning_curve
    raises ValueError when dataset is None
    (presumably because `vak prep` was not run yet)
    """
    root_results_dir = tmp_path.joinpath("test_learncurve_root_results_dir")
    root_results_dir.mkdir()

    keys_to_change = [
        {
            "table": "learncurve",
            "key": "root_results_dir",
            "value": str(root_results_dir),
        },
        {
            "table": "learncurve",
            "key": "dataset",
            "value": "DELETE-KEY"},
    ]

    toml_path = specific_config_toml_path(
        config_type="learncurve",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        keys_to_change=keys_to_change,
    )

    with pytest.raises(KeyError):
        vak.cli.learncurve.learning_curve(toml_path)
