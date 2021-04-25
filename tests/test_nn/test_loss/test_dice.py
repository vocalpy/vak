"""test loss functions"""
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import vak.nn.loss


def tensor_to_gradcheck_var(tensor, dtype=torch.float64, requires_grad=True):
    """Converts the input tensor to a valid variable to check the gradient.
    `gradcheck` needs 64-bit floating point and requires gradient.
    """
    assert torch.is_tensor(tensor), type(tensor)
    return tensor.requires_grad_(requires_grad).type(dtype)


# adapted from kornia, https://github.com/kornia/kornia/blob/master/test/test_losses.py
class TestDiceLoss:
    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 20, device=device, dtype=dtype)
        labels = torch.rand(2, 20) * num_classes
        labels = labels.to(device).long()

        criterion = vak.nn.loss.DiceLoss()
        assert criterion(logits, labels) is not None

    def test_all_zeros(self, device, dtype):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 20, device=device, dtype=dtype)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 20, device=device, dtype=torch.int64)

        criterion = vak.nn.loss.DiceLoss()
        loss = criterion(logits, labels)
        assert_allclose(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 20, device=device, dtype=dtype)
        labels = torch.rand(2, 20) * num_classes
        labels = labels.to(device).long()

        logits = tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(vak.nn.dice_loss, (logits, labels), raise_exception=True)

    def test_jit(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 20, device=device, dtype=dtype)
        labels = torch.rand(2, 20) * num_classes
        labels = labels.to(device).long()

        op = vak.nn.dice_loss
        op_script = torch.jit.script(op)

        assert_allclose(op(logits, labels), op_script(logits, labels))

    def test_module(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 20, device=device, dtype=dtype)
        labels = torch.rand(2, 20) * num_classes
        labels = labels.to(device).long()

        op = vak.nn.dice_loss
        op_module = vak.nn.loss.DiceLoss()

        assert_allclose(op(logits, labels), op_module(logits, labels))
