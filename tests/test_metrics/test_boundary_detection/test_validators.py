import torch
import pytest

import vak.metrics.boundary_detection.validators


@pytest.mark.parametrize(
    'y',
    [
        torch.tensor([1, 2, 3]),
        torch.tensor([1.0, 2.0, 3.0]),
    ]
)
def test_is_1d_tensor(y):
    assert vak.metrics.boundary_detection.validators.is_1d_tensor(y) is True


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
        vak.metrics.boundary_detection.validators.is_1d_tensor(y)


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
        vak.metrics.boundary_detection.validators.is_1d_tensor(y)


@pytest.mark.parametrize(
    'y',
    [
        torch.tensor([1, 2, 3]),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([[1, 2, 3], [1, 2, 3]]),
        torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    ]
)
def test_is_1d_or_2d_tensor(y):
    assert vak.metrics.boundary_detection.validators.is_1d_or_2d_tensor(y) is True


@pytest.mark.parametrize(
    'y',
    [
        [1, 2, 3],
        (1, 2, 3),
        [1.0, 2.0, 3.0],
        (1, 2, 3),
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
    ]
)
def test_is_1d_or_2d_tensor_raises_type_error(y):
    with pytest.raises(TypeError):
        vak.metrics.boundary_detection.validators.is_1d_or_2d_tensor(y)


@pytest.mark.parametrize(
    'y',
    [
        # 0d
        torch.tensor(1),
        torch.tensor(1.0),
        # 3d
        torch.tensor([[[1, 2, 3]]]),
        torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]),
    ]
)
def test_is_1d_or_2d_tensor_raises_value_error(y):
    with pytest.raises(ValueError):
        vak.metrics.boundary_detection.validators.is_1d_or_2d_tensor(y)


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
    assert vak.metrics.boundary_detection.validators.is_non_negative(boundary_times)


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
        vak.metrics.boundary_detection.validators.is_non_negative(boundary_times)


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
    assert vak.metrics.boundary_detection.validators.is_strictly_increasing(boundary_times)


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
        vak.metrics.boundary_detection.validators.is_strictly_increasing(boundary_times)


@pytest.mark.parametrize(
    'arr1, arr2',
    [
        (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])),
        (torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])),
    ]
)
def test_have_same_dtype(arr1, arr2):
    assert vak.metrics.boundary_detection.validators.have_same_dtype(arr1, arr2) is True


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
        vak.metrics.boundary_detection.validators.have_same_dtype(arr1, arr2)


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
    vak.metrics.boundary_detection.validators.have_same_ndim(arr1, arr2)


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
        vak.metrics.boundary_detection.validators.have_same_ndim(arr1, arr2)
