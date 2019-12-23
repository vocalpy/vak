from configparser import ConfigParser, NoSectionError
from pathlib import Path

from .. import models


def map_from_config(config_obj, model_names):
    """map a list of model names to model configuration sections from a
    config.ini file.

    Given a ConfigParser instance and a list of model names, returns dict that
    maps model names to config sections. If no section in the config.ini file
    matches the model name, the name will map the value None.

    Can be used to get configuration for only the models specified
    in a certain section of config.ini file, e.g. in the TRAIN section.

    The returned model-config map can be used with vak.models.from_model_config_map
    (and the number of classes and input shape) to get a list of model instances
    ready for training. Any model name that maps to None will use the default
    values defined for the config.

    Parameters
    ----------
    config_obj : configparser.ConfigParser
        instance of ConfigParser with config.ini file already read into it
        that has sections representing configurations for models
    model_names : list
        of str, i.e. names of models specified by a section
        (such as TRAIN or PREDICT) that should each have corresponding sections
        specifying their configuration: hyperparameters such as learning rate, etc.

    Returns
    -------
    model_config_map : dict
        where each key is the name of a model and the corresponding value is
        a section from a config.ini file.
    """
    # load entry points within function, not at module level,
    # to avoid circular dependencies
    # (user would be unable to import models in other packages
    # if the module in the other package needed to `import vak`)
    MODELS = {model_name: model_builder for model_name, model_builder in models.find()}
    MODEL_CONFIG_PARSERS = {model_name: model_config_parser
                            for model_name, model_config_parser in models.find_config_parsers()}

    for model_name in model_names:
        if model_name not in MODELS:
            raise ValueError(
                f'Model not installed: {model_name}. Installed models are: {list(MODELS.keys())}'
            )
        if model_name not in MODEL_CONFIG_PARSERS:
            raise ValueError(
                f'Could not config parser for model: {model_name}. '
                f'Installed model config parsers are: {list(MODEL_CONFIG_PARSERS.keys())}'
            )

    sections = config_obj.sections()
    model_config_map = {}
    for model_name in model_names:
        if model_name in sections:
            model_section_dict = dict(config_obj[model_name].items())
            model_config_map[model_name] = model_section_dict
        else:
            model_config_map[model_name] = {}  # will use default config
    return model_config_map


def map_from_path(config_path, model_names):
    """map a list of model names to model configuration sections from a
    config.ini file.

    Convenience function that wraps config.models.map and accepts
    path to config.ini file (instead of an instance of a ConfigParser
    with the config.ini file already loaded into it)

    Parameters
    ----------
    config_path : str, Path
        to config.ini file
     model_names : list
        of str, i.e. names of models specified by a section
        (such as TRAIN or PREDICT) that should each have corresponding sections
        specifying their configuration: hyperparameters such as learning rate, etc.

    Returns
    -------
    model_config_map : dict
        where each key is the name of a model and the corresponding value is
        a section from a config.ini file.
    """
    # check config_path is a file,
    # because if it doesn't ConfigParser will just return an "empty" instance w/out sections or options
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f'path not recognized as a file: {config_path}')

    config_obj = ConfigParser()
    config_obj.read(config_path)
    return map_from_config(config_obj, model_names)
