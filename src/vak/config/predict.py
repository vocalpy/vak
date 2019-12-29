"""parses [PREDICT] section of config"""
from configparser import NoOptionError
import os

import attr
from attr.validators import instance_of, optional

from .validators import is_a_directory, is_a_file
from .. import models


@attr.s
class PredictConfig:
    """class that represents [PREDICT] section of config.ini file

    Attributes
    ----------
    csv_path : str
        path to where dataset was saved as a csv.
    checkpoint_path : str
        path to directory with checkpoint files saved by Tensorflow, to reload model
    models : list
        of model names. e.g., 'models = TweetyNet, GRUNet, ConvNet'
    spect_scaler_path : str
        path to a saved SpectScaler object used to normalize spectrograms.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.
    """
    csv_path = attr.ib(validator=[instance_of(str), is_a_file])
    checkpoint_path = attr.ib(validator=[instance_of(str), is_a_directory])
    networks = attr.ib()

    spect_scaler_path = attr.ib(validator=optional([instance_of(str), is_a_file]),
                                default=None)


def parse_predict_config(config_obj, config_file):
    """parse [PREDICT] section of config.ini file

    Parameters
    ----------
    config : ConfigParser
        containing config.ini file already loaded by parse function

    Returns
    -------
    predict_config : vak.config.predict.PredictConfig
        instance of PredictConfig class that represents [PREDICT] section
        of config.ini file
    """
    config_dict = {}

    try:
        config_dict['predict_vds_path'] = os.path.expanduser(
            config_obj['PREDICT']['predict_vds_path']
        )
    except KeyError:
        raise KeyError("'predict_vds_path' option not found in [PREDICT] section of "
                            "config.ini file. Please add this option.")

    try:
        config_dict['train_vds_path'] = os.path.expanduser(
            config_obj['PREDICT']['train_vds_path']
        )
    except KeyError:
        raise KeyError("'train_vds_path' option not found in [PREDICT] section of "
                            "config.ini file. Please add this option.")

    try:
        config_dict['checkpoint_path'] = config_obj['PREDICT']['checkpoint_path']
    except KeyError:
        raise KeyError('must specify checkpoint_path in [PREDICT] section '
                       'of config.ini file')

    # load entry points within function, not at module level,
    # to avoid circular dependencies
    MODEL_NAMES = [model_name for model_name, model_builder in models.find()]
    try:
        model_names = [model_name
                       for model_name in config_obj['PREDICT']['models'].split(',')]
    except NoOptionError:
        raise KeyError("'models' option not found in [PREDICT] section of config.ini file. "
                       "Please add this option as a comma-separated list of model names, e.g.:\n"
                       "models = TweetyNet, GRUnet, convnet")
    for model_name in model_names:
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Model {model_name} not found when importing installed models.'
            )
    config_dict['models'] = model_names

    if config_obj.has_option('PREDICT', 'spect_scaler_path'):
        config_dict['spect_scaler_path'] = config_obj['PREDICT']['spect_scaler_path']

    return PredictConfig(**config_dict)
