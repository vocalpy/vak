"""Constants used by :mod:`vak.prep`.

Defined in a separate module to minimize circular imports.
"""
from . import (
    dimensionality_reduction,
    frame_classification
)


VALID_PURPOSES = frozenset(
    [
        "eval",
        "learncurve",
        "predict",
        "train",
    ]
)

INPUT_TYPES = {'audio', 'spect'}

DATASET_TYPE_FUNCTION_MAP = {
    'frame classification': frame_classification.prep_frame_classification_dataset,
    'dimensionality reduction': dimensionality_reduction.prep_dimensionality_reduction_dataset,
}

DATASET_TYPES = tuple(DATASET_TYPE_FUNCTION_MAP.keys())
