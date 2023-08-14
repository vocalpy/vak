"""Helper function that gets default transforms for a model."""
from __future__ import annotations

from ... import models
from . import frame_classification, parametric_umap


def get_default_transform(
    model_name: str,
    mode: str,
    transform_kwargs: dict,
):
    """Get default transforms for a model,
    according to its family and what mode
    the model is being used in.

    Parameters
    ----------
    model_name : str
        Name of model.
    mode : str
        one of {'train', 'eval', 'predict'}. Determines set of transforms.

    Returns
    -------
    transform, target_transform : callable
        one or more vak transforms to be applied to inputs x and, during training, the target y.
        If more than one transform, they are combined into an instance of torchvision.transforms.Compose.
        Note that when mode is 'predict', the target transform is None.
    """
    try:
        model_family = models.registry.MODEL_FAMILY_FROM_NAME[model_name]
    except KeyError as e:
        raise ValueError(
            f"No model family found for the model name specified: {model_name}"
        ) from e

    if model_family == "FrameClassificationModel":
        return frame_classification.get_default_frame_classification_transform(
            mode, transform_kwargs
        )

    elif model_family == "ParametricUMAPModel":
        return parametric_umap.get_default_parametric_umap_transform(
            transform_kwargs
        )
