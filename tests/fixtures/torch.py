"""fixtures for testing ``torch``-specific functionality"""
# adapted from kornia, https://github.com/kornia/kornia/blob/3606cf9c3d1eb3aabd65ca36a0e7cb98944c01ba/conftest.py
from typing import Dict

import pytest
import torch


def get_test_dtypes() -> Dict[str, torch.dtype]:
    """Creates a dictionary with the dtypes the source code.
    Return:
        dict(str, torch.dtype): list with dtype names.
    """
    dtypes: Dict[str, torch.dtype] = {}
    dtypes["float16"] = torch.float16
    dtypes["float32"] = torch.float32
    dtypes["float64"] = torch.float64
    return dtypes


TEST_DTYPES: Dict[str, torch.dtype] = get_test_dtypes()


@pytest.fixture()
def dtype(dtype_name) -> torch.dtype:
    return TEST_DTYPES[dtype_name]
