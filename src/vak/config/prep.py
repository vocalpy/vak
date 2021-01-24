"""parses [PREP] section of config"""
import attr
from attr import converters, validators
from attr.validators import instance_of

from .validators import is_a_directory, is_a_file, is_audio_format, is_annot_format, is_spect_format
from ..converters import expanded_user_path, labelset_to_set


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
            f'invalid type for {attribute} of {instance}: {type(value)}. Type should be float or int.'
        )

    if value == -1:  # specifies "use the remainder of the dataset"
        # so it is valid, but other negative values are not
        return

    if not value >= 0:
        raise ValueError(
            f'value specified for {attribute} of {instance} must be greater than or equal to zero, was {value}'
        )


@attr.s
class PrepConfig:
    """class to represent [PREP] section of config.toml file

    Attributes
    ----------
    data_dir : str
        path to directory with files from which to make dataset
    output_dir : str
        Path to location where data sets should be saved. Default is None,
        in which case data sets are saved in the current working directory.
    audio_format : str
        format of audio files. One of {'wav', 'cbin'}.
    spect_format : str
        format of files containg spectrograms as 2-d matrices.
        One of {'mat', 'npy'}.
    spect_output_dir : str
        path to directory where array files containing spectrograms
        should be saved, when generated from audio files.
        Default is None, in which case the spectrogram files
        are saved in ``data_dir`` by ``vak.io.dataframe.from_files``.
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
    train_dur : float
        total duration of training set, in seconds. When creating a learning curve,
        training subsets of shorter duration (specified by the 'train_set_durs' option
        in the LEARNCURVE section of a config.toml file) will be drawn from this set.
    val_dur : float
        total duration of validation set, in seconds.
    test_dur : float
        total duration of test set, in seconds.
    """
    data_dir = attr.ib(converter=expanded_user_path, validator=is_a_directory)
    output_dir = attr.ib(converter=expanded_user_path, validator=is_a_directory)

    audio_format = attr.ib(validator=validators.optional(is_audio_format), default=None)
    spect_format = attr.ib(validator=validators.optional(is_spect_format), default=None)
    spect_output_dir = attr.ib(converter=converters.optional(expanded_user_path),
                               validator=validators.optional(is_a_directory),
                               default=None)
    annot_file = attr.ib(converter=converters.optional(expanded_user_path),
                         validator=validators.optional(is_a_file), default=None)
    annot_format = attr.ib(validator=validators.optional(is_annot_format), default=None)

    labelset = attr.ib(converter=converters.optional(labelset_to_set),
                       validator=validators.optional(instance_of(set)),
                       default=None)

    train_dur = attr.ib(converter=converters.optional(duration_from_toml_value),
                        validator=validators.optional(is_valid_duration),
                        default=None)
    val_dur = attr.ib(converter=converters.optional(duration_from_toml_value),
                      validator=validators.optional(is_valid_duration),
                      default=None)
    test_dur = attr.ib(converter=converters.optional(duration_from_toml_value),
                       validator=validators.optional(is_valid_duration),
                       default=None)


REQUIRED_PREP_OPTIONS = [
    'data_dir',
    'output_dir',
]


def parse_prep_config(config_toml, config_path):
    """parse [PREP] section of config.toml file

    Parameters
    ----------
    config_toml : dict
        containing configuration file in TOML format, already loaded by parse function
    toml_path : Path
        path to a configuration file in TOML format (used for error messages)

    Returns
    -------
    prep_config : vak.config.prep.PrepConfig
        instance of class that represents [PREP] section of config.toml file
    """
    prep_section = config_toml['PREP']

    if 'spect_format' in prep_section and 'audio_format' in prep_section:
        raise ValueError("[PREP] section of config.toml file cannot specify both audio_format and "
                         "spect_format, unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms")

    if 'spect_format' not in prep_section and 'audio_format' not in prep_section:
        raise ValueError("[PREP] section of config.toml file must specify either audio_format or "
                         "spect_format")

    for required_option in REQUIRED_PREP_OPTIONS:
        if required_option not in prep_section:
            raise KeyError(
                f"the '{required_option}' option is required but was not found in the "
                f"PREP section of the config.toml file: {config_path}"
            )

    return PrepConfig(**prep_section)
