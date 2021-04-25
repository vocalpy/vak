import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import functional as vakF

__all__ = ["dice_loss", "DiceLoss"]


# adapted from kornia (https://github.com/kornia/kornia/blob/master/kornia/losses/dice.py)
# originally based on https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
def dice_loss(
    input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    r"""Criterion that computes Sørensen-Dice Coefficient loss,
    adapted to work on a 1-dimensional time series.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:
    .. math::
        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}
    Where:
       - :math:`X` scores that estimator assigns to each class.
       - :math:`Y` true class labels.

    the loss, is finally computed as:
    .. math::
        \text{loss}(x, class) = 1 - \text{Dice}(x, class)
    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Args:
        input (torch.Tensor): logits tensor with shape :math:`(N, C, T)` where C = number of classes,
          and T = number of timebins
        labels (torch.Tensor): labels tensor with shape :math:`(N, T)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`. Converted to one-hot vector with shape :math:`(N, C, T)`
          to compute the loss.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.
    Return:
        torch.Tensor: the computed loss.
    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 20, requires_grad=True)
        >>> target = torch.empty(1, 20, dtype=torch.long).random_(N)
        >>> output = dice_loss(input, target)
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) == 3:
        raise ValueError(
            "Invalid input shape, should be 3 dimensions (N, C, T). "
            f"Got: {input.shape}"
        )

    if not input.shape[-1:] == target.shape[-1:]:
        raise ValueError(
            "Last dimension of input and target shapes must be the same size. "
            f"Got: {input.shape} and {target.shape}"
        )

    if not input.device == target.device:
        raise ValueError(
            f"input and target must be in the same device. Got: {input.device} and {target.device}"
        )

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = vakF.one_hot(
        target, num_classes=input.shape[1], device=input.device, dtype=input.dtype
    )

    # compute the actual dice score
    dims = (1, 2)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2.0 * intersection / (cardinality + eps)

    return torch.mean(-dice_score + 1.0)


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss,
    adapted to work on a 1-dimensional time series.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:
    .. math::
        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}
    Where:
       - :math:`X` scores that estimator assigns to each class.
       - :math:`Y` true class labels.
    the loss, is finally computed as:
    .. math::
        \text{loss}(x, class) = 1 - \text{Dice}(x, class)
    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Args:
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.
    Shape:
        input (torch.Tensor): logits tensor with shape :math:`(N, C, T)` where C = number of classes,
          and T = number of timebins
        labels (torch.Tensor): labels tensor with shape :math:`(N, T)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.
    Example:
        >>> N = 5  # num_classes
        >>> criterion = DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(input, target, self.eps)
