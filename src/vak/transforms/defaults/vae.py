"""Default transforms for VAE models."""
from __future__ import annotations

import torchvision.transforms

from .. import transforms as vak_transforms


def get_default_vae_transform(
    transform_kwargs,
) -> torchvision.transforms.Compose:
    """Get default transform for VAE model.

    Parameters
    ----------
    transform_kwargs : dict

    Returns
    -------
    transform : Callable
    """
    transforms = [
        vak_transforms.ToFloatTensor(),
        vak_transforms.AddChannel(),
    ]
    return torchvision.transforms.Compose(transforms)
