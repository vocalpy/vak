import pytest


@pytest.fixture
def events_path(generated_results_data_root):
    events_paths = sorted(
        generated_results_data_root.joinpath("train").glob("**/*events*")
    )
    assert len(events_paths) > 0
    return events_paths[0]
