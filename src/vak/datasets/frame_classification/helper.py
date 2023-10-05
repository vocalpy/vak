"""Helper functions used with frame classification datasets."""
from __future__ import annotations

from . import constants


def sample_ids_array_filename_for_subset(subset: str) -> str:
    """Returns name of sample IDs array file for a subset of the training data."""
    return constants.SAMPLE_IDS_ARRAY_FILENAME.replace(
                '.npy', f'-{subset}.npy'
            )


def inds_in_sample_array_filename_for_subset(subset: str) -> str:
    """Returns name of inds in sample array file for a subset of the training data."""
    return constants.INDS_IN_SAMPLE_ARRAY_FILENAME.replace(
        '.npy', f'-{subset}.npy'
    )
