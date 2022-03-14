import pytest

import torch

import vak.nn.functional as F


# adapted from kornia, https://github.com/kornia/kornia/blob/master/test/utils/test_one_hot.py
def test_onehot():
    num_classes = 4
    labels = torch.zeros(2, 2, 1, dtype=torch.int64)
    labels[0, 0, 0] = 0
    labels[0, 1, 0] = 1
    labels[1, 0, 0] = 2
    labels[1, 1, 0] = 3

    # convert labels to one hot tensor
    one_hot = F.one_hot(labels, num_classes)

    assert 1.0 == pytest.approx(one_hot[0, labels[0, 0, 0], 0, 0].item())
    assert 1.0 == pytest.approx(one_hot[0, labels[0, 1, 0], 1, 0].item())
    assert 1.0 == pytest.approx(one_hot[1, labels[1, 0, 0], 0, 0].item())
    assert 1.0 == pytest.approx(one_hot[1, labels[1, 1, 0], 1, 0].item())
