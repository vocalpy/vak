"""Constants used by :mod:`vak.prep`.

Defined in a separate module to minimize circular imports.
"""
VALID_PURPOSES = frozenset(
    [
        "eval",
        "learncurve",
        "predict",
        "train",
    ]
)
