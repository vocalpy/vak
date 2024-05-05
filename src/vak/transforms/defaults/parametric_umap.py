"""Default transforms for Parametric UMAP models."""

from __future__ import annotations

import torchvision.transforms

from .. import transforms as vak_transforms


def get_default_parametric_umap_transform(
    transform_kwargs: dict | None = None,
) -> torchvision.transforms.Compose:
    """Get default transform for frame classification model.

    Parameters
    ----------
    transform_kwargs : dict, optional
        Keyword arguments for transform class.
        Default is None.

    Returns
    -------
    transform : Callable
    """
    if transform_kwargs is None:
        transform_kwargs = {}
    transforms = [
        vak_transforms.ToFloatTensor(),
        vak_transforms.AddChannel(),
    ]
    return torchvision.transforms.Compose(transforms)
