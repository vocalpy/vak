"""Constants used by :mod:`vak.prep`.

Defined in a separate module to minimize circular imports.
"""
from . import frame_classification


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
    'frame classification': frame_classification.prep,
}

DATASET_TYPES = tuple(DATASET_TYPE_FUNCTION_MAP.keys())
