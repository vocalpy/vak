import subprocess
from unittest import mock

import pytest

import vak


DUMMY_CONFIGFILE = './configs/config_2018-12-17.toml'


@pytest.mark.parametrize(
    'args_list',
    [
        ['prep', DUMMY_CONFIGFILE],
        ['train', DUMMY_CONFIGFILE],
        ['learncurve', DUMMY_CONFIGFILE],
        ['eval', DUMMY_CONFIGFILE],
        ['predict', DUMMY_CONFIGFILE],
        ['configfile', 'train', '--add-prep', '--dst', DUMMY_CONFIGFILE]
    ]
)
def test_main(args_list):
    """Test that :func:`vak.__main__.main` calls the function we expect through :func:`vak.cli.cli`"""
    command = args_list[0]
    mock_cli_function = mock.Mock(name=f'mock_{command}')
    with mock.patch.dict(
        vak.cli.cli.CLI_COMMAND_FUNCTION_MAP, {command: mock_cli_function}
    ):
        # we can't do this with `subprocess` since the function won't be mocked in the subprocess,
        # so we need to test indirectly with `arg_list` passed into `main`
        vak.__main__.main(args_list)
        mock_cli_function.assert_called()


def test___main__prints_help_with_no_args(parser, capsys):
    """Test that if we don't pass in any args, we get """
    parser.print_help()
    expected_output = capsys.readouterr().out.rstrip()

    # doing this by calling a `subprocess` is slow but lets us test the CLI directly
    result = subprocess.run("vak", capture_output=True, text=True)  # call `vak` at CLI with no help
    output = result.stdout.rstrip()
     
    assert output == expected_output
