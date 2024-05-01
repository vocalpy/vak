import attr
from attr.validators import instance_of, optional

from .eval import EvalConfig
from .learncurve import LearncurveConfig
from .predict import PredictConfig
from .prep import PrepConfig
from .spect_params import SpectParamsConfig
from .train import TrainConfig


@attr.s
class Config:
    """class to represent config.toml file

    Attributes
    ----------
    prep : vak.config.prep.PrepConfig
        represents ``[vak.prep]`` table of config.toml file
    spect_params : vak.config.spect_params.SpectParamsConfig
        represents ``[SPECT_PARAMS]`` table of config.toml file
    train : vak.config.train.TrainConfig
        represents ``[vak.train]`` table of config.toml file
    eval : vak.config.eval.EvalConfig
        represents ``[vak.eval]`` table of config.toml file
    predict : vak.config.predict.PredictConfig
        represents ``[vak.predict]`` table of config.toml file.
    learncurve : vak.config.learncurve.LearncurveConfig
        represents ``[vak.learncurve]`` table of config.toml file
    """

    spect_params = attr.ib(
        validator=instance_of(SpectParamsConfig), default=SpectParamsConfig()
    )
    prep = attr.ib(validator=optional(instance_of(PrepConfig)), default=None)
    train = attr.ib(validator=optional(instance_of(TrainConfig)), default=None)
    eval = attr.ib(validator=optional(instance_of(EvalConfig)), default=None)
    predict = attr.ib(
        validator=optional(instance_of(PredictConfig)), default=None
    )
    learncurve = attr.ib(
        validator=optional(instance_of(LearncurveConfig)), default=None
    )
