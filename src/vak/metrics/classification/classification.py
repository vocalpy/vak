from . import functional as F


__all__ = [
    'Accuracy',
]


class Accuracy:
    """standard supervised learning classification accuracy.


    Parameters
    ----------
    y_pred : torch.Tensor
    y_true : torch.Tensor

    Returns
    -------
    acc : float
        between 0 and 1. Sum of predicted labels that equal true labels,
        divided by number of true labels.
    """
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        return F.accuracy(y_pred, y_true)
