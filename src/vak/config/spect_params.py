"""parses [SPECT_PARAMS] section of config"""
import attr
from attr import converters, validators
from attr.validators import instance_of


def freq_cutoffs_validator(instance, attribute, value):
    if len(value) != 2:
        raise ValueError(
            f'freq_cutoffs should be a list of two elements, but instead got: {value}')
    if value[0] > value[1]:
        raise ValueError(
            f'lower freq_cutoff should be less than higher freq_cutoff, instead of: {value}'
        )


VALID_TRANSFORM_TYPES = {'log_spect', 'log_spect_plus_one'}


def is_valid_transform_type(instance, attribute, value):
    if value not in VALID_TRANSFORM_TYPES:
        raise ValueError(
            f'Value for `transform_type`, {value}, in [SPECT_PARAMS] '
            'section of .toml file is not recognized. Must be one '
            f'of the following: {VALID_TRANSFORM_TYPES}'
        )


@attr.s
class SpectParamsConfig:
    """represents parameters for making spectrograms from audio and saving in files

    Attributes
    ----------
    fft_size : int
        size of window for Fast Fourier transform, number of time bins. Default is 512.
    step_size : int
        step size for Fast Fourier transform. Default is 64.
    freq_cutoffs : tuple
        of two elements, lower and higher frequencies. Used to bandpass filter audio
        (using a Butter filter) before generating spectrogram.
        Default is None, in which case no bandpass filtering is applied.
    transform_type : str
        one of {'log_spect', 'log_spect_plus_one'}.
        'log_spect' transforms the spectrogram to log(spectrogram), and
        'log_spect_plus_one' does the same thing but adds one to each element.
        Default is None. If None, no transform is applied.
    thresh: int
        threshold minimum power for log spectrogram.
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    freqbins_key : str
        key for accessing vector of frequency bins in files. Default is 'f'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    audio_path_key : str
        key for accessing path to source audio file for spectogram in files.
        Default is 'audio_path'.
    """
    fft_size = attr.ib(converter=int, validator=instance_of(int), default=512)
    step_size = attr.ib(converter=int, validator=instance_of(int), default=64)
    freq_cutoffs = attr.ib(validator=validators.optional(freq_cutoffs_validator),
                           default=None)
    thresh = attr.ib(converter=converters.optional(float),
                     validator=validators.optional(instance_of(float)),
                     default=None)
    transform_type = attr.ib(validator=validators.optional([instance_of(str), is_valid_transform_type]),
                             default=None)
    spect_key = attr.ib(validator=instance_of(str), default='s')
    freqbins_key = attr.ib(validator=instance_of(str), default='f')
    timebins_key = attr.ib(validator=instance_of(str), default='t')
    audio_path_key = attr.ib(validator=instance_of(str), default='audio_path')


def parse_spect_params_config(config_toml, toml_path):
    """parse [SPECT_PARAMS] section of config.toml file

    Parameters
    ----------
    config_toml : dict
        containing configuration file in TOML format, already loaded by parse function
    toml_path : Path
        path to a configuration file in TOML format (used for error messages)

    Returns
    -------
    spect_params_config : vak.config.spect_params.SpectParamsConfig
        instance with attributes set to values specified by config.toml section
        or to defaults.
    """
    # return defaults if config doesn't have SPECT_PARAMS section
    spect_params_section = {}
    if 'SPECT_PARAMS' in config_toml:
        spect_params_section.update(
            config_toml['SPECT_PARAMS'].items()
        )
    return SpectParamsConfig(**spect_params_section)
