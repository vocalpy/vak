"""validators used by attrs-based classes and by vak.parse.parse_config"""
from configparser import ConfigParser
from pathlib import Path

from scipy.io import wavfile
import crowsetta.formats
from ..evfuncs import load_cbin

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
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Model {model_name} not found when importing installed models.'
            )


AUDIO_FORMAT_FUNC_MAP = {
    'cbin': load_cbin,
    'wav': wavfile.read
}
VALID_AUDIO_FORMATS = list(AUDIO_FORMAT_FUNC_MAP.keys())
def is_audio_format(instance, attribute, value):
    """check if valid audio format"""
    if value not in VALID_AUDIO_FORMATS:
        raise ValueError(
            f'{value} is not a valid format for audio files'
        )


VALID_ANNOT_FORMATS = crowsetta.formats._INSTALLED
def is_annot_format(instance, attribute, value):
    """check if valid annotation format"""
    if value not in VALID_ANNOT_FORMATS:
        raise ValueError(
            f'{value} is not a valid format for annotation files.\n'
            f'Valid formats are: {VALID_ANNOT_FORMATS}'
        )


VALID_SPECT_FORMATS = {'mat', 'npz'}
def is_spect_format(instance, attribute, value):
    """check if valid format for spectrograms"""
    if value not in VALID_SPECT_FORMATS:
        raise ValueError(
            f'{value} is not a valid format for spectrogram files.\n'
            f'Valid formats are: {VALID_SPECT_FORMATS}'
        )


CONFIG_DIR = Path(__file__).parent
VALID_INI_PATH = CONFIG_DIR.joinpath('valid.ini')
VALID_INI = ConfigParser()
VALID_INI.read(VALID_INI_PATH)
VALID_SECTIONS = VALID_INI.sections()
VALID_OPTIONS = {
    section: VALID_INI.options(section)
    for section in VALID_SECTIONS
}


def are_sections_valid(user_config_parser, user_config_path):
    user_sections = user_config_parser.sections()
    for section in user_sections:
        if section not in VALID_SECTIONS:
            raise ValueError(
                f'section defined in {user_config_path} is not '
                f'valid: {section}'
            )


def are_options_valid(user_config_parser, section, user_config_path):
    user_options = set(user_config_parser.options(section))
    valid_options = set(VALID_OPTIONS[section])
    if not user_options.issubset(valid_options):
        invalid_options = user_options - valid_options
        raise ValueError(
            f"the following options from {section} section in "
            f"{user_config_path} are not valid: {invalid_options}"
        )
