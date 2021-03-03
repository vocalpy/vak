import pytest

from vak.constants import RESULTS_DIR_PREFIX


@pytest.fixture
def previous_run_path(generated_test_data_root):
    learncurve_results_root = generated_test_data_root.joinpath(
        'results/learncurve/audio_cbin_annot_notmat'
    )
    results_dirs = sorted(learncurve_results_root.glob(f'{RESULTS_DIR_PREFIX}*'))
    assert len(results_dirs) >= 1
    return results_dirs[-1]
