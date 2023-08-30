import pytest

# this constant defined "by hand" declares all the models for which there are fixtures,
# instead of computing something dynamically e.g. from ``vak.models``.
# Should be used throughout fixtures when we need to get things "by model"
MODELS = [
    "TweetyNet",
]


@pytest.fixture
def default_model():
    """default model used whenever a model is needed to run a test.
    Should work regardless of where the test is run, i.e. both on
    CI platform and locally.

    currently ``TweetyNet``
    """
    return "TweetyNet"
