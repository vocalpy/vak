import attr
from attr.validators import instance_of, optional

from .eval import EvalConfig
from .learncurve import LearncurveConfig
from .predict import PredictConfig
from .prep import PrepConfig
from .train import TrainConfig


@attr.s
class Config:
    """Class that represents a configuration file.

    Attributes
    ----------
    prep : vak.config.prep.PrepConfig
        Represents ``[vak.prep]`` table of config.toml file
    train : vak.config.train.TrainConfig
        Represents ``[vak.train]`` table of config.toml file
    eval : vak.config.eval.EvalConfig
        Represents ``[vak.eval]`` table of config.toml file
    predict : vak.config.predict.PredictConfig
        Represents ``[vak.predict]`` table of config.toml file.
    learncurve : vak.config.learncurve.LearncurveConfig
        Represents ``[vak.learncurve]`` table of config.toml file
    """
    prep = attr.ib(validator=optional(instance_of(PrepConfig)), default=None)
    train = attr.ib(validator=optional(instance_of(TrainConfig)), default=None)
    eval = attr.ib(validator=optional(instance_of(EvalConfig)), default=None)
    predict = attr.ib(
        validator=optional(instance_of(PredictConfig)), default=None
    )
    learncurve = attr.ib(
        validator=optional(instance_of(LearncurveConfig)), default=None
    )
