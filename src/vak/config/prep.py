"""Class and functions for ``[vak.prep]`` table of configuration file."""

from __future__ import annotations

import inspect

import dask.bag
from attrs import converters, define, field, validators
from attrs.validators import instance_of

from .. import prep
from ..common.converters import expanded_user_path, labelset_to_set
from .spect_params import SpectParamsConfig
from .validators import is_annot_format, is_audio_format, is_spect_format


def duration_from_toml_value(value):
    """converter for dataset split durations.
    If value is -1, that value is returned -- specifies "use the remainder of the dataset".
    Other values are converted to float when possible."""
    if value == -1:
        return value
    else:
        return float(value)


def is_valid_duration(instance, attribute, value):
    """validator for dataset split durations"""
    if type(value) not in {int, float}:
        raise TypeError(
            f"invalid type for {attribute} of {instance}: {type(value)}. Type should be float or int."
        )

    if value == -1:  # specifies "use the remainder of the dataset"
        # so it is valid, but other negative values are not
        return

    if not value >= 0:
        raise ValueError(
            f"value specified for {attribute} of {instance} must be greater than or equal to zero, was {value}"
        )


def are_valid_dask_bag_kwargs(instance, attribute, value):
    """validator for ``audio_dask_bag_kwargs``"""
    if not isinstance(value, dict):
        raise TypeError(
            f"Option ``audio_dask_bag_kwargs`` should be a dict but was a {type(value)}.\n"
            "So that it parses as a dict, please specify this option "
            "as an inline table in the .toml file, e.g.\n"
            "`audio_dask_bag_kwargs = { npartitions = 20}`"
        )
    kwargs = list(value.keys())
    valid_bag_kwargs = list(
        inspect.signature(dask.bag.from_sequence).parameters.keys()
    )
    if not all([kwarg in valid_bag_kwargs for kwarg in kwargs]):
        invalid_kwargs = [
            kwarg for kwarg in kwargs if kwarg not in valid_bag_kwargs
        ]
        print(
            f"Invalid keyword arguments specified in ``audio_dask_bag_kwargs``: {invalid_kwargs}"
        )


REQUIRED_KEYS = (
    "data_dir",
    "output_dir",
)


@define
class PrepConfig:
    """Class that represents ``[vak.prep]`` table of configuration file.

    Attributes
    ----------
    data_dir : str
        path to directory with files from which to make dataset
    output_dir : str
        Path to location where data sets should be saved. Default is None,
        in which case data sets are saved in the current working directory.
    dataset_type : str
        String name of the type of dataset, e.g.,
        'frame_classification'. Dataset types are
        defined by machine learning tasks, e.g.,
        a 'frame_classification' dataset would be used
        a :class:`vak.models.FrameClassificationModel` model.
        Valid dataset types are defined as
        :const:`vak.prep.prep.DATASET_TYPES`.
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
    spect_format : str
        format of files containg spectrograms as 2-d matrices.
        One of {'mat', 'npy'}.
    spect_params: vak.config.SpectParamsConfig, optional
        Parameters for Short-Time Fourier Transform and post-processing
        of spectrograms.
        Instance of :class:`vak.config.SpectParamsConfig` class.
        Optional, default is None.
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid.
    annot_file : str
        Path to a single annotation file. Default is None.
        Used when a single file contains annotations for multiple audio files.
    labelset : set
        of str or int, the set of labels that correspond to annotated segments
        that a network should learn to segment and classify. Note that if there
        are segments that are not annotated, e.g. silent gaps between songbird
        syllables, then `vak` will assign a dummy label to those segments
        -- you don't have to give them a label here.
        Value for ``labelset`` is converted to a Python ``set``
        using ``vak.config.converters.labelset_from_toml_value``.
        See help for that function for details on how to specify labelset.
    audio_dask_bag_kwargs : dict
        Keyword arguments used when calling ``dask.bag.from_sequence``
        inside ``vak.io.audio``, where it is used to parallelize
        the conversion of audio files into spectrograms.
        Option should be specified in config.toml file as an inline table,
        e.g., ``audio_dask_bag_kwargs = { npartitions = 20 }``.
        Allows for finer-grained control
        when needed to process files of different sizes.
    train_dur : float
        total duration of training set, in seconds. When creating a learning curve,
        training subsets of shorter duration (specified by the 'train_set_durs' option
        in the LEARNCURVE section of a config.toml file) will be drawn from this set.
    val_dur : float
        total duration of validation set, in seconds.
    test_dur : float
        total duration of test set, in seconds.
    train_set_durs : list, optional
        Durations of datasets to use for a learning curve.
        Float values, durations in seconds of subsets taken from training data
        to create a learning curve, e.g. [5., 10., 15., 20.]. Default is None.
        Required if config file has a learncurve section.
    num_replicates : int, optional
        Number of replicates to train for each training set duration
        in a learning curve. Each replicate uses a different
        randomly drawn subset of the training data (but of the same duration).
        Default is None. Required if config file has a learncurve section.
    """

    data_dir = field(converter=expanded_user_path)
    output_dir = field(converter=expanded_user_path)

    dataset_type = field(validator=instance_of(str))

    @dataset_type.validator
    def is_valid_dataset_type(self, attribute, value):
        if value not in prep.constants.DATASET_TYPES:
            raise ValueError(
                f"Invalid dataset type: {value}. "
                f"Valid dataset types are: {prep.constants.DATASET_TYPES}"
            )

    input_type = field(validator=instance_of(str))

    @input_type.validator
    def is_valid_input_type(self, attribute, value):
        if value not in prep.constants.INPUT_TYPES:
            raise ValueError(
                f"Invalid input type: {value}. Must be one of: {prep.constants.INPUT_TYPES}"
            )

    audio_format = field(
        validator=validators.optional(is_audio_format), default=None
    )
    spect_format = field(
        validator=validators.optional(is_spect_format), default=None
    )
    spect_params = field(
        validator=validators.optional(instance_of(SpectParamsConfig)),
        default=None,
    )
    annot_file = field(
        converter=converters.optional(expanded_user_path),
        default=None,
    )
    annot_format = field(
        validator=validators.optional(is_annot_format), default=None
    )

    labelset = field(
        converter=converters.optional(labelset_to_set),
        validator=validators.optional(instance_of(set)),
        default=None,
    )

    audio_dask_bag_kwargs = field(
        validator=validators.optional(are_valid_dask_bag_kwargs), default=None
    )

    train_dur = field(
        converter=converters.optional(duration_from_toml_value),
        validator=validators.optional(is_valid_duration),
        default=None,
    )
    val_dur = field(
        converter=converters.optional(duration_from_toml_value),
        validator=validators.optional(is_valid_duration),
        default=None,
    )
    test_dur = field(
        converter=converters.optional(duration_from_toml_value),
        validator=validators.optional(is_valid_duration),
        default=None,
    )
    train_set_durs = field(
        validator=validators.optional(instance_of(list)), default=None
    )
    num_replicates = field(
        validator=validators.optional(instance_of(int)), default=None
    )

    def __attrs_post_init__(self):
        if self.audio_format is not None and self.spect_format is not None:
            raise ValueError("cannot specify audio_format and spect_format")

        if self.audio_format is None and self.spect_format is None:
            raise ValueError(
                "must specify either audio_format or spect_format"
            )

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> PrepConfig:
        """Return :class:`PrepConfig` instance from a :class:`dict`.

        The :class:`dict` passed in should be the one found
        by loading a valid configuration toml file with
        :func:`vak.config.parse.from_toml_path`,
        and then using key ``prep``,
        i.e., ``PrepConfig.from_config_dict(config_dict['prep'])``."""
        for required_key in REQUIRED_KEYS:
            if required_key not in config_dict:
                raise KeyError(
                    "The `[vak.prep]` table in a configuration file requires "
                    f"the key '{required_key}', but it was not found "
                    "when loading the configuration file into a Python dictionary. "
                    "Please check that the configuration file is formatted correctly."
                )
        if "spect_params" in config_dict:
            config_dict["spect_params"] = SpectParamsConfig(
                **config_dict["spect_params"]
            )
        return cls(**config_dict)
