import pathlib
from unittest import mock

import pytest

import vak


@pytest.fixture
def parser():
    return vak.__main__.get_parser()


def test_parser_usage(parser,
                      capsys):
    with pytest.raises(SystemExit):
        parser.parse_args(args=[''])
    captured = capsys.readouterr()
    assert captured.err.startswith(
        "usage: vak [-h] command configfile"
    )


def test_parser_help(parser,
                     capsys):
    with pytest.raises(SystemExit):
        parser.parse_args(['-h'])
    captured = capsys.readouterr()
    assert captured.out.startswith(
        "usage: vak [-h] command configfile"
    )


DUMMY_CONFIGFILE = './configs/config_2018-12-17.toml'


@pytest.mark.parametrize(
    'command, raises',
    [
        ('prep', False),
        ('train', False),
        ('learncurve', False),
        ('eval', False),
        ('predict', False),
        ('not-a-valid-command', True),
    ]
)
def test_parser(command,
                raises,
                parser,
                capsys):
    if raises:
        with pytest.raises(SystemExit):
            parser.parse_args([command, DUMMY_CONFIGFILE])
    else:
        args = parser.parse_args([command, DUMMY_CONFIGFILE])
        assert args.command == command
        assert args.configfile == pathlib.Path(DUMMY_CONFIGFILE)


@pytest.mark.parametrize(
    'command',
    [
        'prep',
        'train',
        'learncurve',
        'eval',
        'predict',
    ]
)
def test_main(command,
              parser):
    args = parser.parse_args([command, DUMMY_CONFIGFILE])
    mock_cli_function = mock.Mock(name=f'mock_{command}')
    with mock.patch.dict(vak.cli.cli.COMMAND_FUNCTION_MAP,
                         {command: mock_cli_function}) as mock_command_function_map:
        vak.__main__.main(args)
        mock_cli_function.assert_called()
