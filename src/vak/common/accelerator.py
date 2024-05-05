import torch


def get_default() -> str:
    """Get default `accelerator` for :class:`lightning.pytorch.Trainer`.

    Returns
    -------
    accelerator : str
        Will be ``'gpu'`` if :func:`torch.cuda.is_available`
        is ``True``, and ``'cpu'`` if not.
    """
    if torch.cuda.is_available():
        return "gpu"
    else:
        return "cpu"
