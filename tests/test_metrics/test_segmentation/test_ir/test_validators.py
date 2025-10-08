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
def test_is_1d_tensor(y):
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
def test_is_1d_tensor_raises_type_error(y):
    with pytest.raises(TypeError):
        vak.metrics.segmentation.ir.validators.is_1d_tensor(y)




@pytest.mark.parametrize(
    'y',
    [
        # 0d
        torch.tensor(1),
        torch.tensor(1.0),
        # 2d
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[1.0, 2.0, 3.0]]),
        # 3d
        torch.tensor([[[1, 2, 3]]]),
        torch.tensor([[[1.0, 2.0, 3.0]]]),
    ]
)
def test_is_1d_tensor_raises_value_error(y):
    with pytest.raises(ValueError):
        vak.metrics.segmentation.ir.validators.is_1d_tensor(y)


@pytest.mark.parametrize(
    'y',
    [
        torch.tensor([1.0, 2.0, 3.0]),
        # ---- edge cases
        # empty arrays are valid boundary arrays,
        # e.g. when we segment but don't find any boundaries
        torch.tensor([], dtype=torch.float32),
        # a single boundary is still a valid boundary array
        # e.g for segmenting algorithms that threshold a distance measure
        torch.tensor([1.0]),
    ]
)
def test_is_valid_boundaries_array(y):
    assert vak.metrics.segmentation.ir.validators.is_valid_boundaries_tensor(y) is True


@pytest.mark.parametrize(
    'y',
    [
        # list of ints
        [1, 2, 3],
        # tuple of ints
        (1, 2, 3),
        # list of float
        [1.0, 2.0, 3.0],
        # tuple of float
        (1, 2, 3),
        # invalid dtype
        torch.tensor([1, 2, 3]),
        torch.tensor([1]),
        torch.tensor([], dtype=torch.int16),
    ]
)
def test_is_valid_boundaries_array_raises_type_error(y):
    with pytest.raises(TypeError):
        vak.metrics.segmentation.ir.validators.is_valid_boundaries_tensor(y)


@pytest.mark.parametrize(
    'y',
    [
        # 0d
        torch.tensor(1),
        torch.tensor(1.0),
        # 2d
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[1.0, 2.0, 3.0]]),
        # 3d
        torch.tensor([[[1, 2, 3]]]),
        torch.tensor([[[1.0, 2.0, 3.0]]]),
        # has negative values
        torch.tensor([[[-1, 0, 2, 3]]]),
        torch.tensor([[[-1.0, 0.0, 1.0, 2.0, 3.0]]]),
        # is not monotonically increasing
        # we need `torch.flip` because torch doesn't allow negative indexing
        # e.g., to reverse with `torch.tensor([[[1, 2, 3]]])[::-1]  # ValueError: step muist be greater than zero`
        # see https://github.com/data-apis/array-api-compat/issues/144
        torch.flip(torch.tensor([[[1, 2, 3]]]), dims=(0,)),
    ]
)
def test_is_valid_boundaries_array_raises_value_error(y):
    with pytest.raises(ValueError):
        vak.metrics.segmentation.ir.validators.is_valid_boundaries_tensor(y)


@pytest.mark.parametrize(
    'arr1, arr2',
    [
        (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])),
        (torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])),
    ]
)
def test_have_same_dtype(arr1, arr2):
    assert vak.metrics.segmentation.ir.validators.have_same_dtype(arr1, arr2) is True



@pytest.mark.parametrize(
    'arr1, arr2',
    [
        # (int, float)
        (torch.tensor([1, 2, 3]), torch.tensor([4., 5., 6.])),
        # (float, int)
        (torch.tensor([1., 2., 3.]),  torch.tensor([4, 5, 6])),
    ]
)
def test_have_same_dtype_raises_value_error(arr1, arr2):
    with pytest.raises(ValueError):
        vak.metrics.segmentation.ir.validators.have_same_dtype(arr1, arr2)
