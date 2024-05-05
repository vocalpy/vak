import pytest
import torch

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture(params=DEVICES)
def device(request):
    """parametrized device function,
    that returns string names of the devices
    that ``torch`` considers "available".

    causes any test using ``device`` fixture to run just once
    if only a cpu is available,
    and twice if ``torch.cuda.is_available()`` returns ``True``."""
    return request.param
