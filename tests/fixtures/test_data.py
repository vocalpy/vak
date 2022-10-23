from pathlib import Path

import pytest

HERE = Path(__file__).parent


TEST_DATA_ROOT = HERE.joinpath("..", "data_for_tests")


@pytest.fixture
def test_data_root():
    """Path that points to root of data_for_tests directory"""
    return TEST_DATA_ROOT


SOURCE_TEST_DATA_ROOT = TEST_DATA_ROOT.joinpath("source")


@pytest.fixture
def source_test_data_root(test_data_root):
    """'source' test data, i.e., files **not** created by vak, that is,
    the input data used when vak does create files (csv files, logs,
    neural network checkpoints, etc.)
    """
    return SOURCE_TEST_DATA_ROOT


GENERATED_TEST_DATA_ROOT = TEST_DATA_ROOT.joinpath("generated")


@pytest.fixture
def generated_test_data_root(test_data_root):
    """'generated' test data, i.e. files created by vak:
    csv files, logs, neural network checkpoints, etc."""
    return GENERATED_TEST_DATA_ROOT


@pytest.fixture
def generated_prep_data_root(generated_test_data_root):
    return GENERATED_TEST_DATA_ROOT / "prep"


@pytest.fixture
def generated_results_data_root(generated_test_data_root):
    return GENERATED_TEST_DATA_ROOT / "results"
