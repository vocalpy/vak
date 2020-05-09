"""validators used by attrs-based classes and by vak.parse.parse_config"""
from pathlib import Path

import toml

from .. import constants
from .. import models


def is_a_directory(instance, attribute, value):
    """check if given path is a directory"""
    if not Path(value).is_dir():
        raise NotADirectoryError(
            f'Value specified for {attribute.name} of {type(instance)} not recognized as a directory:\n'
            f'{value}'
        )


def is_a_file(instance, attribute, value):
    """check if given path is a file"""
    if not Path(value).is_file():
        raise FileNotFoundError(
            f'Value specified for {attribute.name} of {type(instance)} not recognized as a file:\n'
            f'{value}'
        )


def is_valid_model_name(instance, attribute, value):
    MODEL_NAMES = [model_name for model_name, model_builder in models.find()]
    for model_name in value:
        if model_name not in MODEL_NAMES and f'{model_name}Model' not in MODEL_NAMES:
            raise ValueError(
                f'Model {model_name} not found when importing installed models.'
            )


def is_audio_format(instance, attribute, value):
    """check if valid audio format"""
    if value not in constants.VALID_AUDIO_FORMATS:
        raise ValueError(
            f'{value} is not a valid format for audio files'
        )


def is_annot_format(instance, attribute, value):
    """check if valid annotation format"""
    if value not in constants.VALID_ANNOT_FORMATS:
        raise ValueError(
            f'{value} is not a valid format for annotation files.\n'
            f'Valid formats are: {constants.VALID_ANNOT_FORMATS}'
        )


def is_spect_format(instance, attribute, value):
    """check if valid format for spectrograms"""
    if value not in constants.VALID_SPECT_FORMATS:
        raise ValueError(
            f'{value} is not a valid format for spectrogram files.\n'
            f'Valid formats are: {constants.VALID_SPECT_FORMATS}'
        )


CONFIG_DIR = Path(__file__).parent
VALID_TOML_PATH = CONFIG_DIR.joinpath('valid.toml')
with VALID_TOML_PATH.open('r') as fp:
    VALID_DICT = toml.load(fp)
VALID_SECTIONS = list(VALID_DICT.keys())
VALID_OPTIONS = {
    section: list(options.keys())
    for section, options in VALID_DICT.items()
}


def are_sections_valid(config_dict, toml_path):
    sections = list(config_dict.keys())
    MODEL_NAMES = [model_name for model_name, model_builder in models.find()]
    # add model names to valid sections so users can define model config in sections
    valid_sections = VALID_SECTIONS + MODEL_NAMES
    for section in sections:
        if section not in valid_sections and f'{section}Model' not in valid_sections:
            raise ValueError(
                f'section defined in {toml_path} is not valid: {section}'
            )


def are_options_valid(config_dict, section, toml_path):
    user_options = set(config_dict[section].keys())
    valid_options = set(VALID_OPTIONS[section])
    if not user_options.issubset(valid_options):
        invalid_options = user_options - valid_options
        raise ValueError(
            f"the following options from {section} section in "
            f"the config file '{toml_path.name}' are not valid:\n{invalid_options}"
        )
