import attr
from attr.validators import instance_of, optional

from .dataloader import DataLoaderConfig
from .eval import EvalConfig
from .learncurve import LearncurveConfig
from .predict import PredictConfig
from .prep import PrepConfig
from .spect_params import SpectParamsConfig
from .train import TrainConfig
from .train_checkpoint import Train_CheckpointConfig


@attr.s
class Config:
    """class to represent config.toml file

    Attributes
    ----------
    prep : vak.config.prep.PrepConfig
        represents ``[PREP]`` section of config.toml file
    spect_params : vak.config.spect_params.SpectParamsConfig
        represents ``[SPECT_PARAMS]`` section of config.toml file
    dataloader : vak.config.dataloader.DataLoaderConfig
        represents ``[DATALOADER]`` section of config.toml file
    train : vak.config.train.TrainConfig
        represents ``[TRAIN]`` section of config.toml file
    eval : vak.config.eval.EvalConfig
        represents ``[EVAL]`` section of config.toml file
    predict : vak.config.predict.PredictConfig
        represents ``[PREDICT]`` section of config.toml file.
    learncurve : vak.config.learncurve.LearncurveConfig
        represents ``[LEARNCURVE]`` section of config.toml file
    train_checkpoint : vak.config.train_checkpoint.Train_CheckpointConfig
        represents ``[TRAIN_CHECKPOINT]`` section of config.toml file
    """

    spect_params = attr.ib(
        validator=instance_of(SpectParamsConfig), default=SpectParamsConfig()
    )
    dataloader = attr.ib(
        validator=instance_of(DataLoaderConfig), default=DataLoaderConfig()
    )

    prep = attr.ib(validator=optional(instance_of(PrepConfig)), default=None)
    train = attr.ib(validator=optional(instance_of(TrainConfig)), default=None)
    train_checkpoint = attr.ib(validator=optional(instance_of(Train_CheckpointConfig)), default=None)
    eval = attr.ib(validator=optional(instance_of(EvalConfig)), default=None)
    predict = attr.ib(validator=optional(instance_of(PredictConfig)), default=None)
    learncurve = attr.ib(
        validator=optional(instance_of(LearncurveConfig)), default=None
    )
