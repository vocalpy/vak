"""parses [PREDICT] section of config"""
from configparser import NoOptionError

import attr
from attr import converters, validators
from attr.validators import instance_of, optional

from .converters import comma_separated_list, expanded_user_path
from .validators import is_a_directory, is_a_file, is_valid_model_name


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
    # required
    checkpoint_path = attr.ib(converter=expanded_user_path,
                              validator=is_a_directory)
    models = attr.ib(converter=comma_separated_list,
                     validator=[instance_of(list), is_valid_model_name])

    # optional
    # csv_path is actually 'required' but we can't enforce that here because cli.prep looks at
    # what sections are defined to figure out where to add csv_path after it creates the csv
    csv_path = attr.ib(converter=converters.optional(expanded_user_path),
                       validator=validators.optional(is_a_file),
                       default=None
                       )

    spect_scaler_path = attr.ib(validator=optional([instance_of(str), is_a_file]),
                                default=None)


REQUIRED_PREDICT_OPTIONS = [
    'checkpoint_path',
    'models',
]


def parse_predict_config(config_obj, config_path):
    """parse [PREDICT] section of config.ini file

    Parameters
    ----------
    config_obj : ConfigParser
        containing config.ini file already loaded by parse function
    config_path : str
        path to config.ini file (used for error messages)

    Returns
    -------
    predict_config : vak.config.predict.PredictConfig
        instance of PredictConfig class that represents [PREDICT] section
        of config.ini file
    """
    predict_section = dict(
        config_obj['PREDICT'].items()
    )

    for required_option in REQUIRED_PREDICT_OPTIONS:
        if required_option not in predict_section:
            raise NoOptionError(
                f"the '{required_option}' option is required but was not found in the "
                f"PREDICT section of the config.ini file: {config_path}"
            )
    return PredictConfig(**predict_section)
