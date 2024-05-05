"""tests for vak.cli.eval module"""
from unittest import mock

import pytest

import vak.cli.eval
import vak.config
import vak.common.constants
import vak.common.paths

from . import cli_asserts


@pytest.mark.parametrize(
    "model_name, audio_format, spect_format, annot_format",
    [
        ("TweetyNet", "cbin", None, "notmat"),
    ],
)
def test_eval(
    model_name, audio_format, spect_format, annot_format, specific_config_toml_path, tmp_path, trainer_table
):
    output_dir = tmp_path.joinpath(
        f"test_eval_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    keys_to_change = [
        {"table": "eval", "key": "output_dir", "value": str(output_dir)},
        {"table": "eval", "key": "trainer", "value": trainer_table},
    ]

    toml_path = specific_config_toml_path(
        config_type="eval",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        keys_to_change=keys_to_change,
    )

    with mock.patch('vak.eval.eval', autospec=True) as mock_core_eval:
        vak.cli.eval.eval(toml_path)
        assert mock_core_eval.called

    assert cli_asserts.log_file_created(command="eval", output_path=output_dir)

    assert cli_asserts.log_file_contains_version(command="eval", output_path=output_dir)


def test_eval_dataset_none_raises(
        specific_config_toml_path
):
    """Test that cli.eval raises ValueError when dataset is None
    (presumably because `vak prep` was not run yet)
    """
    keys_to_change = [
        {"table": "eval", "key": "dataset", "value": "DELETE-KEY"},
    ]

    toml_path = specific_config_toml_path(
        config_type="eval",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        keys_to_change=keys_to_change,
    )

    with pytest.raises(KeyError):
        vak.cli.eval.eval(toml_path)
