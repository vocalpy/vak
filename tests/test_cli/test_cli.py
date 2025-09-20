import argparse
import pathlib

import pytest

import vak.cli.cli


def test_get_parser():
    """Smoke test that just makes sure we get back a parser as expected"""
    parser = vak.cli.cli.get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_parser_usage(parser,
                      capsys):
    with pytest.raises(SystemExit):
        parser.parse_args(args=[''])
    captured = capsys.readouterr()
    assert captured.err.startswith(
        "usage: vak [-h] {prep,train,eval,predict,learncurve,configfile} ..."
    )


def test_parser_help(parser,
                     capsys):
    with pytest.raises(SystemExit):
        parser.parse_args(['-h'])
    captured = capsys.readouterr()
    assert captured.out.startswith(
        "usage: vak [-h] {prep,train,eval,predict,learncurve,configfile} ..."
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