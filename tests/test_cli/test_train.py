"""tests for vak.cli.train module"""
from unittest import mock

import pytest

import vak.config
import vak.common.constants
import vak.common.paths
import vak.cli.train

from . import cli_asserts


@pytest.mark.parametrize(
    "model_name, audio_format, spect_format, annot_format",
    [
        ("TweetyNet", "cbin", None, "notmat"),
        ("TweetyNet", None, "mat", "yarden"),
    ],
)
def test_train(
    model_name, audio_format, spect_format, annot_format, specific_config_toml_path, tmp_path, trainer_table
):
    root_results_dir = tmp_path.joinpath("test_train_root_results_dir")
    root_results_dir.mkdir()

    keys_to_change = [
        {
            "table": "train",
            "key": "root_results_dir",
            "value": str(root_results_dir),
        },
        {"table": "train", "key": "trainer", "value": trainer_table},
    ]

    toml_path = specific_config_toml_path(
        config_type="train",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        keys_to_change=keys_to_change,
    )

    with mock.patch('vak.train.train', autospec=True) as mock_core_train:
        vak.cli.train.train(toml_path)
        assert mock_core_train.called

    results_path = sorted(root_results_dir.glob(f"{vak.common.constants.RESULTS_DIR_PREFIX}*"))
    assert len(results_path) == 1
    results_path = results_path[0]
    assert cli_asserts.toml_config_file_copied_to_results_path(results_path, toml_path)
    assert cli_asserts.log_file_created(command="train", output_path=results_path)
    assert cli_asserts.log_file_contains_version(command="train", output_path=results_path)


def test_train_dataset_path_none_raises(
        specific_config_toml_path, tmp_path,
):
    """Test that cli.train raises ValueError when dataset_path is None
    (presumably because `vak prep` was not run yet)
    """
    root_results_dir = tmp_path.joinpath("test_train_root_results_dir")
    root_results_dir.mkdir()

    keys_to_change = [
        {"table": "train", "key": "root_results_dir", "value": str(root_results_dir)},
        {"table": "train", "key": "dataset", "value": "DELETE-KEY"},
    ]

    toml_path = specific_config_toml_path(
        config_type="train",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        keys_to_change=keys_to_change,
    )

    with pytest.raises(KeyError):
        vak.cli.train.train(toml_path)
