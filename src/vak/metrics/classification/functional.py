import torch


def accuracy(y_pred, y_true):
    """standard supervised learning classification accuracy:
    Sum of predicted labels that equal true labels, divided by number of true labels.

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
    correct = torch.eq(y_pred, y_true).view(-1)
    return correct.sum().item() / correct.shape[0]
