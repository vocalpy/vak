from . import biosoundsegbench
from .cmacbench import CMACBench, SplitsMetadata
from .get import get

__all__ = [
    "cmacbench",
    "CMACBench",
    "get",
    "SplitsMetadata",
]

# TODO: make this a proper registry
DATASETS = {"CMACBench": CMACBench}
