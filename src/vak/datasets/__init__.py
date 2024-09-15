from . import cmacbench
from .cmacbench import CMACBench
from .get import get

__all__ = [
    "cmacbench",
    "CMACBench",
    "get",
]

# TODO: make this a proper registry
DATASETS = {"CMACBench": CMACBench}
