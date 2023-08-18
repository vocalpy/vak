"""Default transforms for Parametric UMAP models."""
from __future__ import annotations

import torchvision.transforms

from .. import transforms as vak_transforms


def get_default_parametric_umap_transform(
    transform_kwargs,
) -> torchvision.transforms.Compose:
    """Get default transform for frame classification model.

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
