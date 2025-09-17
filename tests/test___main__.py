import pathlib
from unittest import mock

import pytest

import vak


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
