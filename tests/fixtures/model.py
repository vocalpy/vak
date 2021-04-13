import pytest


@pytest.fixture
def default_model():
    """default model used whenever a model is needed to run a test.
    Should work regardless of where the test is run, i.e. both on
    CI platform and locally.

    currently ``teenytweetynet``
    """
    return "teenytweetynet"
