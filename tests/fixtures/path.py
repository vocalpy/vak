import pytest

from hashlib import md5
from time import localtime


from vak.constants import RESULTS_DIR_PREFIX


@pytest.fixture
def previous_run_path(generated_test_data_root):
    learncurve_results_root = generated_test_data_root.joinpath(
        'results/learncurve/audio_cbin_annot_notmat'
    )
    results_dirs = sorted(learncurve_results_root.glob(f'{RESULTS_DIR_PREFIX}*'))
    assert len(results_dirs) >= 1
    return results_dirs[-1]


@pytest.fixture
def random_path_factory(tmp_path):
    """factory function that returns a random path to a file that does not exist"""

    def _random_path(suffix):
        prefix = md5(str(localtime()).encode('utf-8')).hexdigest()
        return tmp_path / f"{prefix}_{suffix}"

    return _random_path
