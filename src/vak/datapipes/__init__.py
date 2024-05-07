"""Module that contains datapipe classes,
used to load inputs and targets for neural network models
from datasets prepared by :func:`vak.prep.prep`"""

from . import frame_classification, parametric_umap

__all__ = ["frame_classification", "parametric_umap"]
