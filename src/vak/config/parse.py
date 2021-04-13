from pathlib import Path

import toml
from toml.decoder import TomlDecodeError

from .config import Config
from .dataloader import DataLoaderConfig
from .eval import EvalConfig
from .learncurve import LearncurveConfig
from .predict import PredictConfig
from .prep import PrepConfig
from .spect_params import SpectParamsConfig
from .train import TrainConfig
from .validators import are_sections_valid, are_options_valid

SECTION_CLASSES = {
    "DATALOADER": DataLoaderConfig,
    "EVAL": EvalConfig,
    "LEARNCURVE": LearncurveConfig,
    "PREDICT": PredictConfig,
    "PREP": PrepConfig,
    "SPECT_PARAMS": SpectParamsConfig,
    "TRAIN": TrainConfig,
}

REQUIRED_OPTIONS = {
    "DATALOADER": None,
    "EVAL": [
        "checkpoint_path",
        "labelmap_path",
        "output_dir",
        "models",
    ],
    "LEARNCURVE": [
        "models",
        "root_results_dir",
        "train_set_durs",
        "num_replicates",
    ],
    "PREDICT": [
        "checkpoint_path",
        "labelmap_path",
        "models",
    ],
    "PREP": [
        "data_dir",
        "output_dir",
    ],
    "SPECT_PARAMS": None,
    "TRAIN": [
        "models",
        "root_results_dir",
    ],
}


def parse_config_section(config_toml, section_name, toml_path=None):
    """parse section of config.toml file

    Parameters
    ----------
    config_toml : dict
        containing config.toml file already loaded by parse function
    section_name : str
        name of section from configuration
        file that should be parsed
    toml_path : str
        path to a configuration file in TOML format. Default is None.
        Used for error messages if specified.

    Returns
    -------
    config : vak.config section class
        instance of class that represents section of config.toml file,
        e.g. PredictConfig for 'PREDICT' section
    """
    section = dict(config_toml[section_name].items())

    required_options = REQUIRED_OPTIONS[section_name]
    if required_options is not None:
        for required_option in required_options:
            if required_option not in section:
                if toml_path:
                    err_msg = (
                        f"the '{required_option}' option is required but was not found in the "
                        f"{section_name} section of the config.toml file: {toml_path}"
                    )
                else:
                    err_msg = (
                        f"the '{required_option}' option is required but was not found in the "
                        f"{section_name} section of the toml config"
                    )
                raise KeyError(err_msg)
    return SECTION_CLASSES[section_name](**section)


def _validate_sections_arg_convert_list(sections):
    if isinstance(sections, str):
        sections = [sections]
    elif isinstance(sections, list):
        if not all([isinstance(section_name, str) for section_name in sections]):
            raise ValueError("all section names in 'sections' should be strings")
        if not all(
            [section_name in list(SECTION_CLASSES.keys()) for section_name in sections]
        ):
            raise ValueError(
                "all section names in 'sections' should be valid names of sections. "
                f"Values for 'sections were: {sections}.\n"
                f"Valid section names are: {list(SECTION_CLASSES.keys())}"
            )
    return sections


def from_toml(config_toml, toml_path=None, sections=None):
    """load a TOML configuration file

    Parameters
    ----------
    config_toml : dict
        Python ``dict`` containing a .toml configuration file,
        parsed by the ``toml`` library.
    toml_path : str, Path
        path to a configuration file in TOML format. Default is None.
        Not required, used only to make any error messages clearer.
    sections : str, list
        name of section or sections from configuration
        file that should be parsed. Can be a string
        (single section) or list of strings (multiple
        sections). Default is None,
        in which case all are validated and parsed.

    Returns
    -------
    config : vak.config.parse.Config
        instance of Config class, whose attributes correspond to
        sections in a config.toml file.
    """
    are_sections_valid(config_toml, toml_path)

    sections = _validate_sections_arg_convert_list(sections)

    config_dict = {}
    if sections is None:
        sections = list(
            SECTION_CLASSES.keys()
        )  # i.e., parse all sections, except model
    for section_name in sections:
        if section_name in config_toml:
            are_options_valid(config_toml, section_name, toml_path)
            config_dict[section_name.lower()] = parse_config_section(
                config_toml, section_name, toml_path
            )

    return Config(**config_dict)


def _load_toml_from_path(toml_path):
    """helper function to load toml config file,
    factored out to use in other modules when needed

    checks if ``toml_path`` exists before opening,
    and tries to give a clear message if an error occurs when parsing"""
    toml_path = Path(toml_path)
    if not toml_path.is_file():
        raise FileNotFoundError(f".toml config file not found: {toml_path}")

    try:
        with toml_path.open("r") as fp:
            config_toml = toml.load(fp)
    except TomlDecodeError as e:
        raise Exception(f"Error when parsing .toml config file: {toml_path}") from e

    return config_toml


def from_toml_path(toml_path, sections=None):
    """parse a TOML configuration file

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format.
        Parsed by ``toml`` library, then converted to an
        instance of ``vak.config.parse.Config`` by
        calling ``vak.parse.from_toml``
    sections : str, list
        name of section or sections from configuration
        file that should be parsed. Can be a string
        (single section) or list of strings (multiple
        sections). Default is None,
        in which case all are validated and parsed.

    Returns
    -------
    config : vak.config.parse.Config
        instance of Config class, whose attributes correspond to
        sections in a config.toml file.
    """
    config_toml = _load_toml_from_path(toml_path)
    return from_toml(config_toml, toml_path, sections)
