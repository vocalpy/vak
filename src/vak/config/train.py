"""Class that represents ``[vak.train]`` table of configuration file."""

from attrs import converters, define, field, validators
from attrs.validators import instance_of

from ..common.converters import bool_from_str, expanded_user_path
from .dataset import DatasetConfig
from .model import ModelConfig
from .trainer import TrainerConfig

REQUIRED_KEYS = (
    "dataset",
    "model",
    "root_results_dir",
    "trainer",
)


@define
class TrainConfig:
    """Class that represents ``[vak.train]`` table of configuration file.

    Attributes
    ----------
    model : vak.config.ModelConfig
        The model to use: its name,
        and the parameters to configure it.
        Must be an instance of :class:`vak.config.ModelConfig`
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    batch_size : int
        number of samples per batch presented to models during training.
    root_results_dir : str
        directory in which results will be created.
        The vak.cli.train function will create
        a subdirectory in this directory each time it runs.
    dataset : vak.config.DatasetConfig
        The dataset to use: the path to it,
        and optionally a path to a file representing splits,
        and the name, if it is a built-in dataset.
        Must be an instance of :class:`vak.config.DatasetConfig`.
    trainer : vak.config.TrainerConfig
        Configuration for :class:`lightning.pytorch.Trainer`.
        Must be an instance of :class:`vak.config.TrainerConfig`.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader.
    shuffle: bool
        if True, shuffle training data before each epoch. Default is True.
    standardize_frames : bool
        if True, use :class:`vak.transforms.FramesStandardizer` to standardize the frames.
        Normalization is done by subtracting off the mean for each row
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    val_step : int
        Step on which to estimate accuracy using validation set.
        If val_step is n, then validation is carried out every time
        the global step / n is a whole number, i.e., when val_step modulo the global step is 0.
        Default is None, in which case no validation is done.
    ckpt_step : int
        Step on which to save to checkpoint file.
        If ckpt_step is n, then a checkpoint is saved every time
        the global step / n is a whole number, i.e., when ckpt_step modulo the global step is 0.
        Default is None, in which case checkpoint is only saved at the last epoch.
    patience : int
        number of validation steps to wait without performance on the
        validation set improving before stopping the training.
        Default is None, in which case training only stops after the specified number of epochs.
    checkpoint_path : str
        path to directory with checkpoint files saved by Torch, to reload model.
        Default is None, in which case a new model is initialized.
    frames_standardizer_path : str
        path to a saved :class:`vak.transforms.FramesStandardizer` object used to standardize (normalize) frames.
        If spectrograms were normalized and this is not provided, will give
        incorrect results. Default is None.
    """

    # required
    model = field(
        validator=instance_of(ModelConfig),
    )
    num_epochs = field(converter=int, validator=instance_of(int))
    batch_size = field(converter=int, validator=instance_of(int))
    root_results_dir = field(converter=expanded_user_path)
    dataset: DatasetConfig = field(
        validator=instance_of(DatasetConfig),
    )
    trainer: TrainerConfig = field(
        validator=instance_of(TrainerConfig),
    )

    results_dirname = field(
        converter=converters.optional(expanded_user_path),
        default=None,
    )

    standardize_frames = field(
        converter=bool_from_str,
        validator=validators.optional(instance_of(bool)),
        default=False,
    )

    num_workers = field(validator=instance_of(int), default=2)
    shuffle = field(
        converter=bool_from_str, validator=instance_of(bool), default=True
    )

    val_step = field(
        converter=converters.optional(int),
        validator=validators.optional(instance_of(int)),
        default=None,
    )
    ckpt_step = field(
        converter=converters.optional(int),
        validator=validators.optional(instance_of(int)),
        default=None,
    )
    patience = field(
        converter=converters.optional(int),
        validator=validators.optional(instance_of(int)),
        default=None,
    )
    checkpoint_path = field(
        converter=converters.optional(expanded_user_path),
        default=None,
    )
    frames_standardizer_path = field(
        converter=converters.optional(expanded_user_path),
        default=None,
    )

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> "TrainConfig":
        """Return :class:`TrainConfig` instance from a :class:`dict`.

        The :class:`dict` passed in should be the one found
        by loading a valid configuration toml file with
        :func:`vak.config.parse.from_toml_path`,
        and then using key ``train``,
        i.e., ``TrainConfig.from_config_dict(config_dict['train'])``."""
        for required_key in REQUIRED_KEYS:
            if required_key not in config_dict:
                raise KeyError(
                    "The `[vak.train]` table in a configuration file requires "
                    f"the option '{required_key}', but it was not found "
                    "when loading the configuration file into a Python dictionary. "
                    "Please check that the configuration file is formatted correctly."
                )
        config_dict["model"] = ModelConfig.from_config_dict(
            config_dict["model"]
        )
        config_dict["dataset"] = DatasetConfig.from_config_dict(
            config_dict["dataset"]
        )
        config_dict["trainer"] = TrainerConfig(**config_dict["trainer"])
        return cls(**config_dict)
