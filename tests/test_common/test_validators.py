import pytest
import torch

import vak.common.validators


@pytest.mark.parametrize(
    'tensor, expected_exception',
    [
        (torch.tensor([0, 1, 2, 3]), None),
        (torch.tensor([0.1, 1,1, 2.1, 3.1]), None),
        # raise type error if not a tensor
        ([0, 1, 2, 3], TypeError),
        ((1, 2, 3), TypeError),
        ([0.1, 1,1, 2.1, 3.1], TypeError),
        # raise a ValueError if not 1-D
        # 0-D
        (torch.tensor(1), ValueError),
        (torch.tensor(1.0), ValueError),
        # 2-D
        (torch.tensor([[0, 1, 2, 3]]), ValueError),
        (torch.tensor([[1, 2, 3]]), ValueError),
        (torch.tensor([[1.0, 2.0, 3.0]]), ValueError),
        (torch.tensor([[0.1, 1,1, 2.1, 3.1]]), ValueError),
        # 3-D
        (torch.tensor([[[1, 2, 3]]]), ValueError),
        (torch.tensor([[[1.0, 2.0, 3.0]]]), ValueError),
    ]
)
def test_is_1d_tensor(tensor, expected_exception):
    if expected_exception is None:
        assert vak.common.validators.is_1d_tensor(tensor)
    else:
        with pytest.raises(expected_exception):
            vak.common.validators.is_1d_tensor(tensor)


@pytest.mark.parametrize(
    'tensor, expected_exception',
    [
        (torch.tensor([[0, 1, 2, 3]]), None),
        (torch.tensor([[0.1, 1,1, 2.1, 3.1]]), None),
        (torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]), None),
        (torch.tensor([[0.1, 1,1, 2.1, 3.1], [0.1, 1,1, 2.1, 3.1]]), None),
        (torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]), None),
        (torch.tensor([[0.1, 1,1, 2.1, 3.1], [0.1, 1,1, 2.1, 3.1], [0.1, 1,1, 2.1, 3.1]]), None),
        # raise type error if not a tensor
        ([0, 1, 2, 3], TypeError),
        ((1, 2, 3), TypeError),
        ([0.1, 1,1, 2.1, 3.1], TypeError),
        # raise a ValueError if not 1-D
        # 0-D
        (torch.tensor(1), ValueError),
        (torch.tensor(1.0), ValueError),
        # 1-D
        (torch.tensor([0, 1, 2, 3]), ValueError),
        (torch.tensor([1, 2, 3]), ValueError),
        (torch.tensor([1.0, 2.0, 3.0]), ValueError),
        (torch.tensor([0.1, 1,1, 2.1, 3.1]), ValueError),
        # 3-D
        (torch.tensor([[[1, 2, 3]]]), ValueError),
        (torch.tensor([[[1.0, 2.0, 3.0]]]), ValueError),
    ]
)
def test_is_2d_tensor(tensor, expected_exception):
    if expected_exception is None:
        assert vak.common.validators.is_2d_tensor(tensor)
    else:
        with pytest.raises(expected_exception):
            vak.common.validators.is_2d_tensor(tensor)


@pytest.mark.parametrize(
    'y, expected_exception',
    [
        (torch.tensor([1, 2, 3]), None),
        (torch.tensor([1.0, 2.0, 3.0]), None),
        (torch.tensor([[1, 2, 3], [1, 2, 3]]), None),
        (torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), None),
        # TypeError if not a tensor
        ([1, 2, 3], TypeError),
        ((1, 2, 3), TypeError),
        ([1.0, 2.0, 3.0], TypeError),
        ((1, 2, 3), TypeError),
        ([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], TypeError),
        # ValueError if not 1-D or 2-D
        # 0d
        (torch.tensor(1), ValueError),
        (torch.tensor(1.0), ValueError),
        # 3d
        (torch.tensor([[[1, 2, 3]]]), ValueError),
        (torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]), ValueError),
    ]
)
def test_is_1d_or_2d_tensor(y, expected_exception):
    if expected_exception is None:
        assert vak.common.validators.is_1d_or_2d_tensor(y)
    else:
        with pytest.raises(expected_exception):
            vak.common.validators.is_1d_or_2d_tensor(y)


@pytest.mark.parametrize(
    'boundary_times',
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
def test_is_non_negative(boundary_times):
    assert vak.common.validators.is_non_negative(boundary_times)


@pytest.mark.parametrize(
    'boundary_times',
    [
        # has negative values
        torch.tensor([-1, 0, 2, 3]),
        torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0]),
    ]
)
def test_is_non_negative_raises_value_error(boundary_times):
    with pytest.raises(ValueError):
        vak.common.validators.is_non_negative(boundary_times)


@pytest.mark.parametrize(
    'boundary_times',
    [
        torch.tensor([1.0, 2.0, 3.0]),
        # ---- edge cases
        # empty arrays are valid boundary arrays,
        # e.g. when we segment but don't find any boundaries
        torch.tensor([], dtype=torch.float32),
        # a single boundary is still a valid boundary array
        # e.g for segmenting algorithms that threshold a distance measure
        torch.tensor([1.0]),
        # even though these have negative values, they are still strictly increasing
        torch.tensor([-1, 0, 2, 3]),
        torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0]),
    ]
)
def test_is_strictly_increasing(boundary_times):
    assert vak.common.validators.is_strictly_increasing(boundary_times)


@pytest.mark.parametrize(
    'boundary_times',
    [
        torch.tensor([1.0, 3.0, 2.0]),
        torch.tensor([2, 3, -1, 0]),
        torch.tensor([1.0, 2.0, 3.0, -1.0, 0.0]),
    ]
)
def test_is_strictly_increasing_raises_value_error(boundary_times):
    with pytest.raises(ValueError):
        vak.common.validators.is_strictly_increasing(boundary_times)


@pytest.mark.parametrize(
    'arr1, arr2',
    [
        (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])),
        (torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])),
    ]
)
def test_have_same_dtype(arr1, arr2):
    assert vak.common.validators.have_same_dtype(arr1, arr2) is True


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
        vak.common.validators.have_same_dtype(arr1, arr2)


@pytest.mark.parametrize(
    'arr1, arr2',
    [
        (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])),
        (torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])),
        (torch.tensor([[1., 2., 3.]]), torch.tensor([[4., 5., 6.]])),
        (torch.tensor([[[1., 2., 3.]]]), torch.tensor([[[4., 5., 6.]]])),
    ]
)
def test_have_same_ndim(arr1, arr2):
    vak.common.validators.have_same_ndim(arr1, arr2)


@pytest.mark.parametrize(
    'arr1, arr2',
    [
        (torch.tensor(1), torch.tensor([4, 5, 6])),
        (torch.tensor(1.0), torch.tensor([4., 5., 6.])),
        (torch.tensor([1., 2., 3]), torch.tensor([[4., 5., 6.]])),
        (torch.tensor([1., 2., 3.]), torch.tensor([[[4., 5., 6.]]])),
    ]
)
def test_have_same_ndim_raises(arr1, arr2):
    with pytest.raises(ValueError):
        vak.common.validators.have_same_ndim(arr1, arr2)
