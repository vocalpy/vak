"""Class that represents ``[vak.predict]`` table of configuration file."""

from __future__ import annotations

import os
from pathlib import Path

from attr import converters, validators
from attr.validators import instance_of
from attrs import define, field

from ..common.converters import expanded_user_path
from .dataset import DatasetConfig
from .model import ModelConfig
from .trainer import TrainerConfig

REQUIRED_KEYS = (
    "checkpoint_path",
    "dataset",
    "model",
    "trainer",
)


@define
class PredictConfig:
    """Class that represents ``[vak.predict]`` table of configuration file.

     Attributes
     ----------
    checkpoint_path : str
        path to directory with checkpoint files saved by Torch, to reload model
    labelmap_path : str
        path to 'labelmap.json' file.
    model : vak.config.ModelConfig
        The model to use: its name,
        and the parameters to configure it.
        Must be an instance of :class:`vak.config.ModelConfig`
    batch_size : int
        number of samples per batch presented to models during training.
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
        Argument to torch.DataLoader. Default is 2.
    frames_standardizer_path : str
        path to a saved :class:`vak.transforms.FramesStandardizer` object used to standardize (normalize) frames.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.
    annot_csv_filename : str
        name of .csv file containing predicted annotations.
        Default is None, in which case the name of the dataset .csv
        is used, with '.annot.csv' appended to it.
    output_dir : str
        path to location where .csv containing predicted annotation
        should be saved. Defaults to current working directory.
    min_segment_dur : float
        minimum duration of segment, in seconds. If specified, then
        any segment with a duration less than min_segment_dur is
        removed from lbl_tb. Default is None, in which case no
        segments are removed.
    majority_vote : bool
        if True, transform segments containing multiple labels
        into segments with a single label by taking a "majority vote",
        i.e. assign all time bins in the segment the most frequently
        occurring label in the segment. This transform can only be
        applied if the labelmap contains an 'unlabeled' label,
        because unlabeled segments makes it possible to identify
        the labeled segments. Default is False.
    save_net_outputs : bool
        If True, save 'raw' outputs of neural networks
        before they are converted to annotations. Default is False.
        Typically the output will be "logits"
        to which a softmax transform might be applied.
        For each item in the dataset--each row in  the `dataset_path` .csv--
        the output will be saved in a separate file in `output_dir`,
        with the extension `{MODEL_NAME}.output.npz`. E.g., if the input is a
        spectrogram with `spect_path` filename `gy6or6_032312_081416.npz`,
        and the network is `TweetyNet`, then the net output file
        will be `gy6or6_032312_081416.tweetynet.output.npz`.
    """

    # required, external files
    checkpoint_path = field(converter=expanded_user_path)
    labelmap_path = field(converter=expanded_user_path)

    # required, model / dataloader
    model = field(
        validator=instance_of(ModelConfig),
    )
    batch_size = field(converter=int, validator=instance_of(int))
    dataset: DatasetConfig = field(
        validator=instance_of(DatasetConfig),
    )
    trainer: TrainerConfig = field(
        validator=instance_of(TrainerConfig),
    )

    # optional, transform
    frames_standardizer_path = field(
        converter=converters.optional(expanded_user_path),
        default=None,
    )

    # optional, data loader
    num_workers = field(validator=instance_of(int), default=2)

    annot_csv_filename = field(
        validator=validators.optional(instance_of(str)), default=None
    )
    output_dir = field(
        converter=expanded_user_path,
        default=Path(os.getcwd()),
    )
    min_segment_dur = field(
        validator=validators.optional(instance_of(float)), default=None
    )
    majority_vote = field(validator=instance_of(bool), default=True)
    save_net_outputs = field(validator=instance_of(bool), default=False)

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> PredictConfig:
        """Return :class:`PredictConfig` instance from a :class:`dict`.

        The :class:`dict` passed in should be the one found
        by loading a valid configuration toml file with
        :func:`vak.config.parse.from_toml_path`,
        and then using key ``predict``,
        i.e., ``PredictConfig.from_config_dict(config_dict['predict'])``."""
        for required_key in REQUIRED_KEYS:
            if required_key not in config_dict:
                raise KeyError(
                    "The `[vak.predict]` table in a configuration file requires "
                    f"the option '{required_key}', but it was not found "
                    "when loading the configuration file into a Python dictionary. "
                    "Please check that the configuration file is formatted correctly."
                )
        config_dict["dataset"] = DatasetConfig.from_config_dict(
            config_dict["dataset"]
        )
        config_dict["model"] = ModelConfig.from_config_dict(
            config_dict["model"]
        )
        config_dict["trainer"] = TrainerConfig(**config_dict["trainer"])
        return cls(**config_dict)
