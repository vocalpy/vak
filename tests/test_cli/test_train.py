"""tests for vak.cli.train module"""
from unittest import mock

import pytest

import vak.config
import vak.constants
import vak.paths
import vak.cli.train

from . import cli_asserts


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

    with mock.patch('vak.core.train', autospec=True) as mock_core_train:
        vak.cli.train.train(toml_path)
        assert mock_core_train.called

    results_path = sorted(root_results_dir.glob(f"{vak.constants.RESULTS_DIR_PREFIX}*"))
    assert len(results_path) == 1
    results_path = results_path[0]
    assert cli_asserts.toml_config_file_copied_to_results_path(results_path, toml_path)
    assert cli_asserts.log_file_created(command="train", output_path=results_path)
    assert cli_asserts.log_file_contains_version(command="train", output_path=results_path)


def test_train_csv_path_none_raises(
        specific_config, tmp_path,
):
    """Test that cli.train raises ValueError when csv_path is None
    (presumably because `vak prep` was not run yet)
    """
    root_results_dir = tmp_path.joinpath("test_train_root_results_dir")
    root_results_dir.mkdir()

    options_to_change = [
        {"section": "TRAIN", "option": "root_results_dir", "value": str(root_results_dir)},
        {"section": "TRAIN", "option": "csv_path", "value": "DELETE-OPTION"},
    ]

    toml_path = specific_config(
        config_type="train",
        model="teenytweetynet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=options_to_change,
    )

    with pytest.raises(ValueError):
        vak.cli.train.train(toml_path)


@pytest.mark.parametrize(
    "config_type, audio_format, spect_format, annot_format",
    [
        ("train", "cbin", None, "notmat"),
        ("train", "wav", None, "birdsong-recognition-dataset"),
        ("train", None, "mat", "yarden"),
        ("train_continue", "cbin", None, "notmat"),
        ("train_continue", "wav", None, "birdsong-recognition-dataset"),
        ("train_continue", None, "mat", "yarden"),
    ],
)
def test_train_passes_correct_labelset_and_labelmap_path(
    config_type, audio_format, spect_format, annot_format, specific_config, tmp_path, model, device
):
    """Test that ``cli.train`` passes in the expected arguments to ``core.train``,
    depending on whether ``labelmap_path`` is specified in the [TRAIN] section
    of the config file or not.
    """
    root_results_dir = tmp_path.joinpath("test_train_root_results_dir")
    root_results_dir.mkdir()

    options_to_change = [
        {"section": "TRAIN", "option": "root_results_dir", "value": str(root_results_dir)},
        {"section": "TRAIN", "option": "device", "value": device},
    ]

    toml_path = specific_config(
        config_type=config_type,
        model=model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )

    cfg = vak.config.parse.from_toml_path(toml_path)

    with mock.patch('vak.core.train', autospec=True) as mock_core_train:
        vak.cli.train.train(toml_path)
        if config_type == "train":
            assert mock_core_train.call_args[1]['labelset'] == cfg.prep.labelset
            assert mock_core_train.call_args[1]['labelmap_path'] is None
        elif config_type == "train_continue":
            assert mock_core_train.call_args[1]['labelset'] is None
            assert mock_core_train.call_args[1]['labelmap_path'] == cfg.train.labelmap_path
