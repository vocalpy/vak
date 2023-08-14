"""validators used by attrs-based classes and by vak.parse.parse_config"""
from pathlib import Path

import toml

from .. import models
from ..common import constants


def is_a_directory(instance, attribute, value):
    """check if given path is a directory"""
    if not Path(value).is_dir():
        raise NotADirectoryError(
            f"Value specified for {attribute.name} of {type(instance)} not recognized as a directory:\n"
            f"{value}"
        )


def is_a_file(instance, attribute, value):
    """check if given path is a file"""
    if not Path(value).is_file():
        raise FileNotFoundError(
            f"Value specified for {attribute.name} of {type(instance)} not recognized as a file:\n"
            f"{value}"
        )


def is_valid_model_name(instance, attribute, value: str) -> None:
    """Validate model name."""
    if value not in models.registry.MODEL_NAMES:
        raise ValueError(
            f"Invalid model name: {value}.\nValid model names are: {models.registry.MODEL_NAMES}"
        )


def is_audio_format(instance, attribute, value):
    """check if valid audio format"""
    if value not in constants.VALID_AUDIO_FORMATS:
        raise ValueError(f"{value} is not a valid format for audio files")


def is_annot_format(instance, attribute, value):
    """check if valid annotation format"""
    if value not in constants.VALID_ANNOT_FORMATS:
        raise ValueError(
            f"{value} is not a valid format for annotation files.\n"
            f"Valid formats are: {constants.VALID_ANNOT_FORMATS}"
        )


def is_spect_format(instance, attribute, value):
    """check if valid format for spectrograms"""
    if value not in constants.VALID_SPECT_FORMATS:
        raise ValueError(
            f"{value} is not a valid format for spectrogram files.\n"
            f"Valid formats are: {constants.VALID_SPECT_FORMATS}"
        )


CONFIG_DIR = Path(__file__).parent
VALID_TOML_PATH = CONFIG_DIR.joinpath("valid.toml")
with VALID_TOML_PATH.open("r") as fp:
    VALID_DICT = toml.load(fp)
VALID_SECTIONS = list(VALID_DICT.keys())
VALID_OPTIONS = {
    section: list(options.keys()) for section, options in VALID_DICT.items()
}


def are_sections_valid(config_dict, toml_path=None):
    sections = list(config_dict.keys())
    from ..cli.cli import CLI_COMMANDS  # avoid circular import

    cli_commands_besides_prep = [
        command for command in CLI_COMMANDS if command != "prep"
    ]
    sections_that_are_commands_besides_prep = [
        section
        for section in sections
        if section.lower() in cli_commands_besides_prep
    ]
    if len(sections_that_are_commands_besides_prep) == 0:
        raise ValueError(
            "did not find a section related to a vak command in config besides `prep`.\n"
            f"Sections in config were: {sections}"
        )

    if len(sections_that_are_commands_besides_prep) > 1:
        raise ValueError(
            "found multiple sections related to a vak command in config besides `prep`.\n"
            f"Those sections are: {sections_that_are_commands_besides_prep}. "
            f"Please use just one command besides `prep` per .toml configuration file"
        )

    MODEL_NAMES = list(models.registry.MODEL_NAMES)
    # add model names to valid sections so users can define model config in sections
    valid_sections = VALID_SECTIONS + MODEL_NAMES
    for section in sections:
        if (
            section not in valid_sections
            and f"{section}Model" not in valid_sections
        ):
            if toml_path:
                err_msg = (
                    f"section defined in {toml_path} is not valid: {section}"
                )
            else:
                err_msg = (
                    f"section defined in toml config is not valid: {section}"
                )
            raise ValueError(err_msg)


def are_options_valid(config_dict, section, toml_path=None):
    user_options = set(config_dict[section].keys())
    valid_options = set(VALID_OPTIONS[section])
    if not user_options.issubset(valid_options):
        invalid_options = user_options - valid_options
        if toml_path:
            err_msg = (
                f"the following options from {section} section in "
                f"the config file '{toml_path.name}' are not valid:\n{invalid_options}"
            )
        else:
            err_msg = (
                f"the following options from {section} section in "
                f"the toml config are not valid:\n{invalid_options}"
            )
        raise ValueError(err_msg)
