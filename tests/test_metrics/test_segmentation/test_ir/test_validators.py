import torch
import pytest

import vak.metrics.segmentation.ir.validators


@pytest.mark.parametrize(
    'y',
    [
        torch.tensor([1, 2, 3]),
        torch.tensor([1.0, 2.0, 3.0]),
    ]
)
def test_is_1d_ndarray(y):
    assert vak.metrics.segmentation.ir.validators.is_1d_tensor(y) is True


@pytest.mark.parametrize(
    'y',
    [
        [1, 2, 3],
        (1, 2, 3),
        [1.0, 2.0, 3.0],
        (1, 2, 3),
    ]
)
def test_is_1d_ndarray_raises_type_error(y):
    with pytest.raises(TypeError):
        vak.metrics.segmentation.ir.validators.is_1d_tensor(y)

