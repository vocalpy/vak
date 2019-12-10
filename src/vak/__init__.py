from .__about__ import (
    __author__,
    __commit__,
    __copyright__,
    __email__,
    __license__,
    __summary__,
    __title__,
    __uri__,
    __version__,
)


from . import __main__
from . import cli
from . import config
from . import core
from . import curvefit
from . import io
from . import evfuncs
from . import koumura_utils
from . import metrics
from . import network
from . import utils

from .io import mat
from .io import Dataset, MetaSpect, Vocalization
