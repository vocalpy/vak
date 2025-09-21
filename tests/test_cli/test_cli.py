import argparse
import pathlib
from unittest import mock

import pytest

import vak.cli.cli


DUMMY_CONFIGFILE_STR = './configs/config_2018-12-17.toml'
DUMMY_CONFIGFILE_PATH = pathlib.Path(DUMMY_CONFIGFILE_STR)


@pytest.mark.parametrize(
    'args_list, expected_attributes',
    [
        (
            ['prep', DUMMY_CONFIGFILE_STR],
            dict(command="prep", configfile=DUMMY_CONFIGFILE_PATH)
        ),
        (
            ['train', DUMMY_CONFIGFILE_STR],
            dict(command="train", configfile=DUMMY_CONFIGFILE_PATH)
        ),
        (
            ['learncurve', DUMMY_CONFIGFILE_STR],
            dict(command="learncurve", configfile=DUMMY_CONFIGFILE_PATH)
        ),
        (
            ['eval', DUMMY_CONFIGFILE_STR],
            dict(command="eval", configfile=DUMMY_CONFIGFILE_PATH)
        ),
        (
            ['predict', DUMMY_CONFIGFILE_STR],
            dict(command="predict", configfile=DUMMY_CONFIGFILE_PATH)
        ),
        (
            ['configfile', 'train'],
            dict(command="configfile", kind="train", add_prep=False, dst=pathlib.Path.cwd())
        ),
        (
            ['configfile', 'eval'],
            dict(command="configfile", kind="eval", add_prep=False, dst=pathlib.Path.cwd())
        ),
        (
            ['configfile', 'train', "--add-prep"],
            dict(command="configfile", kind="train", add_prep=True, dst=pathlib.Path.cwd())
        )
    ]
)
def test_parser_commands_with_configfile(args_list, expected_attributes):
    """Test that calling parser.parse_args gives us a Namespace with the expected args"""
    parser = vak.cli.cli.get_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    args = parser.parse_args(args_list)
    assert isinstance(args, argparse.Namespace)

    for attr_name, expected_value in expected_attributes.items():
        assert hasattr(args, attr_name)
        assert getattr(args, attr_name) == expected_value



def test_parser_raises(parser):
    """Test that an invalid command passed into our ArgumentParser raises a SystemExit"""
    with pytest.raises(SystemExit):
        parser.parse_args(["not-a-valid-command", DUMMY_CONFIGFILE_STR])


@pytest.mark.parametrize(
    'args_list',
    [
        ['prep', DUMMY_CONFIGFILE_STR],
        ['train', DUMMY_CONFIGFILE_STR],
        ['learncurve', DUMMY_CONFIGFILE_STR],
        ['eval', DUMMY_CONFIGFILE_STR],
        ['predict', DUMMY_CONFIGFILE_STR],
        ['configfile', 'train', '--add-prep', '--dst', DUMMY_CONFIGFILE_STR]
    ]
)
def test_cli(
    args_list, parser,    
):
    """Test that :func:`vak.cli.cli.cli` calls the functions we expect"""
    args = parser.parse_args(args_list)

    command = args_list[0]
    mock_cli_function = mock.Mock(name=f'mock_{command}')
    with mock.patch.dict(
        vak.cli.cli.CLI_COMMAND_FUNCTION_MAP, {command: mock_cli_function}
    ):
        # we can't do this with `subprocess` since the function won't be mocked in the subprocess,
        # so we need to test indirectly with `arg_list` passed into `main`
        vak.cli.cli.cli(args)
        mock_cli_function.assert_called()


def test_configfile():
    # FIXME test that configfile works the way we expect
    assert False