"""Default transforms for Parametric UMAP models."""
from __future__ import annotations

from typing import Callable

import torchvision.transforms

from .. import transforms as vak_transforms


def get_default_parametric_umap_transform(transform_kwargs) -> Callable:
    """Get default transform for frame classification model.

    Parameters
    ----------
    transform_kwargs : dict

    Returns
    -------
    transform : Callable
    """
    return torchvision.transforms.Compose(
            [
                vak_transforms.ToFloatTensor(),
                vak_transforms.AddChannel(),
                torchvision.transforms.Resize(transform_kwargs['resize'])
            ]
    )
