"""command-line interface functions for training,
creating learning curves, etc."""

from .cli import cli
from .eval import eval
from .learncurve import learning_curve
from .predict import predict
from .prep import prep
from .train import train

__all__ = [
    'cli',
    'eval',
    'learncurve',
    'predict',
    'prep',
    'train',
]
