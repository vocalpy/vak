import pytest

import vak.cli.cli


@pytest.fixture
def parser():
    """Return an instance of the parser used by the command-line interface,
    by calling :func:`vak.cli.cli.get_parser`"""
    return vak.cli.cli.get_parser()
