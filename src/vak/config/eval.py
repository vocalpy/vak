"""Class and functions for ``[vak.eval]`` table in configuration file."""

from __future__ import annotations

import pathlib

from attrs import converters, define, field, validators
from attrs.validators import instance_of

from ..common.converters import expanded_user_path
from .dataset import DatasetConfig
from .model import ModelConfig
from .trainer import TrainerConfig


def convert_post_tfm_kwargs(post_tfm_kwargs: dict) -> dict:
    post_tfm_kwargs = dict(post_tfm_kwargs)

    if "min_segment_dur" not in post_tfm_kwargs:
        # because there's no null in TOML,
        # users leave arg out of config then we set it to None
        post_tfm_kwargs["min_segment_dur"] = None
    else:
        post_tfm_kwargs["min_segment_dur"] = float(
            post_tfm_kwargs["min_segment_dur"]
        )

    if "majority_vote" not in post_tfm_kwargs:
        # set default for this one too
        post_tfm_kwargs["majority_vote"] = False
    else:
        post_tfm_kwargs["majority_vote"] = bool(
            post_tfm_kwargs["majority_vote"]
        )

    return post_tfm_kwargs


def are_valid_post_tfm_kwargs(instance, attribute, value):
    """check if ``post_tfm_kwargs`` is valid"""
    if not isinstance(value, dict):
        raise TypeError(
            "'post_tfm_kwargs' should be declared in toml config as an inline table "
            f"that parses as a dict, but type was: {type(value)}. "
            "Please declare in a similar fashion: `{majority_vote = True, min_segment_dur = 0.02}`"
        )
    if any(
        [k not in {"majority_vote", "min_segment_dur"} for k in value.keys()]
    ):
        invalid_kwargs = [
            k
            for k in value.keys()
            if k not in {"majority_vote", "min_segment_dur"}
        ]
        raise ValueError(
            f"Invalid keyword argument name specified for 'post_tfm_kwargs': {invalid_kwargs}."
            "Valid names are: {'majority_vote', 'min_segment_dur'}"
        )
    if "majority_vote" in value:
        if not isinstance(value["majority_vote"], bool):
            raise TypeError(
                "'post_tfm_kwargs' keyword argument 'majority_vote' "
                f"should be of type bool but was: {type(value['majority_vote'])}"
            )
    if "min_segment_dur" in value:
        if value["min_segment_dur"] and not isinstance(
            value["min_segment_dur"], float
        ):
            raise TypeError(
                "'post_tfm_kwargs' keyword argument 'min_segment_dur' type "
                f"should be float but was: {type(value['min_segment_dur'])}"
            )


REQUIRED_KEYS = (
    "checkpoint_path",
    "dataset",
    "output_dir",
    "model",
    "trainer",
)


@define
class EvalConfig:
    """Class that represents [vak.eval] table in configuration file.

    Attributes
    ----------
    checkpoint_path : str
        path to directory with checkpoint files saved by Torch, to reload model
    output_dir : str
        Path to location where .csv files with evaluation metrics should be saved.
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
    labelmap_path : str
        path to 'labelmap.json' file.
    frames_standardizer_path : str
        path to a saved :class:`vak.transforms.FramesStandardizer` object used to standardize (normalize) frames.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.
    post_tfm_kwargs : dict
        Keyword arguments to post-processing transform.
        If None, then no additional clean-up is applied
        when transforming labeled timebins to segments,
        the default behavior.
        The transform used is
        ``vak.transforms.frame_labels.PostProcess`.
        Valid keyword argument names are 'majority_vote'
        and 'min_segment_dur', and should be appropriate
        values for those arguments: Boolean for ``majority_vote``,
        a float value for ``min_segment_dur``.
        See the docstring of the transform for more details on
        these arguments and how they work.
    """

    # required, external files
    checkpoint_path: pathlib.Path = field(converter=expanded_user_path)
    output_dir: pathlib.Path = field(converter=expanded_user_path)

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

    # "optional" but actually required for frame classification models
    # TODO: check model family in __post_init__ and raise ValueError if labelmap
    # TODO: not specified for a frame classification model?
    labelmap_path = field(
        converter=converters.optional(expanded_user_path), default=None
    )
    # optional, transform
    frames_standardizer_path = field(
        converter=converters.optional(expanded_user_path),
        default=None,
    )

    post_tfm_kwargs = field(
        validator=validators.optional(are_valid_post_tfm_kwargs),
        converter=converters.optional(convert_post_tfm_kwargs),
        default=None,
    )

    # optional, data loader
    num_workers = field(validator=instance_of(int), default=2)

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> EvalConfig:
        """Return :class:`EvalConfig` instance from a :class:`dict`.

        The :class:`dict` passed in should be the one found
        by loading a valid configuration toml file with
        :func:`vak.config.parse.from_toml_path`,
        and then using key ``eval``,
        i.e., ``EvalConfig.from_config_dict(config_dict['eval'])``."""
        for required_key in REQUIRED_KEYS:
            if required_key not in config_dict:
                raise KeyError(
                    "The `[vak.eval]` table in a configuration file requires "
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
