"""command-line interface functions for training,
creating learning curves, etc."""

from .prep import make_data
from .learncurve import learncurve
from .summary import summary
from .predict import predict
from .cli import cli
from .train import train
