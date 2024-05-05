import pytest
import torch


TRAINER_TABLE = [
    {"accelerator": "cpu", "devices": 1}
]
if torch.cuda.is_available():
    {"accelerator": "gpu", "devices": [0]}


@pytest.fixture(params=TRAINER_TABLE)
def trainer_table(request):
    """Parametrized 'trainer' table for config file

    Causes any test using this :func:`trainer_table` fixture
    to run just once if only a cpu is available,
    and twice if ``torch.cuda.is_available()`` returns ``True``."""
    return request.param
