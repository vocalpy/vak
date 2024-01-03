"""Constants used by :mod:`vak.prep`.

Defined in a separate module to minimize circular imports.
"""
from . import frame_classification, parametric_umap, vae

VALID_PURPOSES = frozenset(
    [
        "eval",
        "learncurve",
        "predict",
        "train",
    ]
)

INPUT_TYPES = {"audio", "spect"}

DATASET_TYPE_FUNCTION_MAP = {
    "frame classification": frame_classification.prep_frame_classification_dataset,
    "parametric umap": parametric_umap.prep_parametric_umap_dataset,
    "vae-window": vae.prep_vae_dataset,
    "vae-segment": vae.prep_vae_dataset,
}

DATASET_TYPES = tuple(DATASET_TYPE_FUNCTION_MAP.keys())
