from . import biosoundsegbench
from .biosoundsegbench import BioSoundSegBench, SplitsMetadata
from .get import get

__all__ = [
    "biosoundsegbench",
    "BioSoundSegBench",
    "get",
    "SplitsMetadata",
]

# TODO: make this a proper registry
DATASETS = {"BioSoundSegBench": BioSoundSegBench}
