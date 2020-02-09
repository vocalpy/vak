"""parses [PREP] section of config"""
import attr
from attr import converters, validators
from attr.validators import instance_of

from .converters import expanded_user_path
from .validators import is_a_directory, is_a_file, is_audio_format, is_annot_format, is_spect_format


def to_set(value):
    tmp_set = set(value)
    if len(tmp_set) == len(value):
        return tmp_set
    elif len(tmp_set) < len(value):
        raise ValueError(
            'Labelset should be set of unique labels for classes applied to segments in annotation, but '
            f'found repeated elements: the input was {value} but the unique set is {tmp_set}'
        )


@attr.s
class PrepConfig:
    """class to represent [PREP] section of config.ini file

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
    train_dur : float
        total duration of training set, in seconds. When creating a learning curve,
        training subsets of shorter duration (specified by the 'train_set_durs' option
        in the LEARNCURVE section of a config.ini file) will be drawn from this set.
    val_dur : float
        total duration of validation set, in seconds.
    test_dur : float
        total duration of test set, in seconds.
    """
    data_dir = attr.ib(converter=expanded_user_path, validator=is_a_directory)
    output_dir = attr.ib(converter=expanded_user_path, validator=is_a_directory)

    audio_format = attr.ib(validator=validators.optional(is_audio_format), default=None)
    spect_format = attr.ib(validator=validators.optional(is_spect_format), default=None)
    annot_file = attr.ib(converter=converters.optional(expanded_user_path),
                         validator=validators.optional(is_a_file), default=None)
    annot_format = attr.ib(validator=validators.optional(is_annot_format), default=None)

    labelset = attr.ib(converter=converters.optional(to_set),
                       validator=validators.optional(instance_of(set)),
                       default=None)

    train_dur = attr.ib(converter=converters.optional(float),
                        validator=validators.optional(instance_of(float)),
                        default=None)
    val_dur = attr.ib(converter=converters.optional(float),
                      validator=validators.optional(instance_of(float)),
                      default=None)
    test_dur = attr.ib(converter=converters.optional(float),
                       validator=validators.optional(instance_of(float)),
                       default=None)


REQUIRED_PREP_OPTIONS = [
    'data_dir',
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
        instance of class that represents [PREP] section of config.ini file
    """
    prep_section = config_toml['PREP']

    if 'spect_format' in prep_section and 'audio_format' in prep_section:
        raise ValueError("[PREP] section of config.ini file cannot specify both audio_format and "
                         "spect_format, unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms")

    if 'spect_format' not in prep_section and 'audio_format' not in prep_section:
        raise ValueError("[PREP] section of config.ini file must specify either audio_format or "
                         "spect_format")

    for required_option in REQUIRED_PREP_OPTIONS:
        if required_option not in prep_section:
            raise KeyError(
                f"the '{required_option}' option is required but was not found in the "
                f"PREP section of the config.toml file: {config_path}"
            )

    return PrepConfig(**prep_section)
