"""Helper function that gets default transforms for a model."""

from __future__ import annotations

from typing import Callable, Literal

from ... import models
from . import frame_classification, parametric_umap


def get_default_transform(
    model_name: str,
    mode: Literal["eval", "predict", "train"],
    transform_kwargs: dict | None = None,
) -> Callable:
    """Get default transform for a model,
    according to its family and what mode
    the model is being used in.

    Parameters
    ----------
    model_name : str
        Name of model.
    mode : str
        One of {'eval', 'predict', 'train'}.

    Returns
    -------
    item_transform : callable
        Transform to be applied to input :math:`x` to a model and,
        during training, the target :math:`y`.
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
