import torch


def get_default():
    """get default device for torch.

    Returns
    -------
    device : str
        'cuda' if torch.cuda.is_available() is True,
        and returns 'cpu' otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
