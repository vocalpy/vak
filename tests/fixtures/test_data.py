from pathlib import Path

import pytest

HERE = Path(__file__).parent


@pytest.fixture
def test_data_root():
    """Path that points to root of data_for_tests directory"""
    return HERE.joinpath("..", "data_for_tests")


@pytest.fixture
def source_test_data_root(test_data_root):
    """'source' test data, i.e., files **not** created by vak, that is,
    the input data used when vak does create files (csv files, logs,
    neural network checkpoints, etc.)
    """
    return test_data_root.joinpath("source")


@pytest.fixture
def generated_test_data_root(test_data_root):
    """'generated' test data, i.e. files created by vak:
    csv files, logs, neural network checkpoints, etc."""
    return test_data_root.joinpath("generated")


@pytest.fixture
def generated_prep_data_root(generated_test_data_root):
    return generated_test_data_root / "prep"


@pytest.fixture
def generated_results_data_root(generated_test_data_root):
    return generated_test_data_root / "results"
