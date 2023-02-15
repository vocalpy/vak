from unittest.mock import patch

import pytest
import torch

import vak.transforms.functional as F


@pytest.mark.parametrize(
    'tensor, expected_result',
    [
        (torch.tensor([0]), False),
        (torch.rand((513, 1000)), True),
        (torch.rand((1, 513, 1000)), True),
        (torch.rand((3, 513, 1000)), True),
    ]
)
def test_is_spect(tensor, expected_result):
    result = F.is_spect(tensor)
    assert result == expected_result


@pytest.mark.parametrize(
    'tensor, expected_result',
    [
        (torch.tensor([0]), ValueError),
        (torch.rand((513, 1000)), None),
        (torch.rand((1, 513, 1000)), None),
        (torch.rand((3, 513, 1000)), None),
    ]
)
def test_validate_spect(tensor, expected_result):
    if expected_result is ValueError:
        with pytest.raises(expected_result):
            F.validate_spect(tensor)
    else:
        assert F.validate_spect(tensor) is expected_result


SPECT_SIZES = [
    (513, 1000),
    (1, 513, 1000),
    (3, 513, 1000),
]
EXPECTED_SIZES = [spect_size[-2:] for spect_size in SPECT_SIZES]
GET_SPECT_SIZE_ARGVALS = [
    (torch.rand(size=spect_size), expected_size)
    for spect_size, expected_size in zip(SPECT_SIZES, EXPECTED_SIZES)
]


@pytest.mark.parametrize(
    'spect, expected_size',
    GET_SPECT_SIZE_ARGVALS,
)
def test_get_spect_size(spect, expected_size):
    size = F.get_spect_size(spect)
    assert size == expected_size


@pytest.mark.parametrize(
    'spect, expected_error',
    [
        (torch.tensor([0]), ValueError),
    ]
)
def test_get_spect_size_raises(spect, expected_error):
    with pytest.raises(expected_error):
        F.get_spect_size(spect)


@pytest.mark.parametrize(
    'spect, window_size, start_ind',
    [
        (torch.rand((513, 1000)), 176, 50),
        (torch.rand((1, 513, 1000)), 176, 50),
        (torch.rand((3, 513, 1000)), 176, 50),
    ]
)
def test_random_window(spect, window_size, start_ind, torch_seed):
    torch.manual_seed(torch_seed)
    window = F.random_window(spect, window_size)
    assert window.shape[-1] == window_size

    # hacky, non-mock way of testing we get consistent result
    torch.manual_seed(torch_seed)
    window_again = F.random_window(spect, window_size)
    torch.all(torch.eq(window, window_again))

    with patch('torch.randint') as patched_randint:
        patched_randint.return_value = torch.tensor([start_ind])
        window_patched = F.random_window(spect, window_size)
        assert torch.all(
            torch.eq(window_patched, spect[..., start_ind: start_ind + window_size])
        )


def test_pad_to_window():
    assert False


def test_standardize_spect():
    assert False


def test_to_floattensor():
    assert False


def test_to_longtensor():
    assert False


def test_view_as_window_batch():
    assert False
