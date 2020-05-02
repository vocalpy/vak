from pathlib import Path

import toml

from .. import models
from ..engine.model import Model


def map_from_config_dict(config_dict, model_names):
    """map a list of model names to model configuration sections from a
    config.toml file.

    Given the configuraiton in a dict and a list of model names, returns dict that
    maps model names to config sections. If no section in the config.toml file
    matches the model name, an error is raised.

    Used to get configuration for only the models specified
    in a certain section of config.toml file, e.g. in the TRAIN section.

    The returned model-config map can be used with vak.models.from_model_config_map
    (and the number of classes and input shape) to get a list of model instances
    ready for training.

    Parameters
    ----------
    config_dict : dict
        configuration from a .toml file, loaded into a dictionary
    model_names : list
        of str, i.e. names of models specified by a section
        (such as TRAIN or PREDICT) that should each have corresponding sections
        specifying their configuration: hyperparameters such as learning rate, etc.

    Returns
    -------
    model_config_map : dict
        where each key is the name of a model and the corresponding value is
        a section from a config.toml file.
    """
    # first check whether models in list are installed

    # load entry points within function, not at module level,
    # to avoid circular dependencies
    # (user would be unable to import models in other packages
    # if the module in the other package needed to `import vak`)
    MODELS = {model_name: model_builder for model_name, model_builder in models.find()}
    for model_name in model_names:
        if model_name not in MODELS:
            # try appending 'Model' to name
            tmp_model_name = f'{model_name}Model'
            if tmp_model_name not in MODELS:
                raise ValueError(
                    f"Did not find an installed model named {model_name} or {tmp_model_name}. "
                    f"Installed models are: {list(MODELS.keys())}"
                )

    # now see if we can find corresponding sections in config
    sections = list(config_dict.keys())
    model_config_map = {}
    for model_name in model_names:
        if model_name in sections:
            model_config_dict = config_dict[model_name]
        else:
            # try appending 'Model' to name
            tmp_model_name = f'{model_name}Model'
            if tmp_model_name not in sections:
                raise ValueError(
                    f'did not find section named {model_name} or {tmp_model_name} '
                    f'in config'
                )
            model_config_dict = config_dict[tmp_model_name]

        # check if config declares parameters for required attributes;
        # if not, just put an empty dict that will get passed as the "kwargs"
        for attr in Model.REQUIRED_SUBCLASS_ATTRIBUTES:
            if attr not in model_config_dict:
                model_config_dict[attr] = {}

        model_config_map[model_name] = config_dict[model_name]

    return model_config_map


def map_from_path(toml_path, model_names):
    """map a list of model names to sections from a .toml configuration file
     that specify parameters for those models.

    Parameters
    ----------
    toml_path : str, Path
        to configuration file in .toml format
     model_names : list
        of str, i.e. names of models specified by a section
        (such as TRAIN or PREDICT) that should each have corresponding sections
        specifying their configuration: hyperparameters such as learning rate, etc.

    Returns
    -------
    model_config_map : dict
        where each key is the name of a model and the corresponding value is
        a section from a config.toml file.
    """
    # check config_path is a file,
    # because if it doesn't ConfigParser will just return an "empty" instance w/out sections or options
    toml_path = Path(toml_path)
    if not toml_path.is_file():
        raise FileNotFoundError(f'file not found, or not recognized as a file: {toml_path}')

    with toml_path.open('r') as fp:
        config_dict = toml.load(fp)
    return map_from_config_dict(config_dict, model_names)
