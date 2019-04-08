"""parses [PREDICT] section of config"""
from configparser import NoOptionError

import attr
from attr.validators import instance_of, optional

from .validators import is_a_directory, is_a_file
from .. import network


@attr.s
class PredictConfig:
    """class that represents [PREDICT] section of config.ini file

    Attributes
    ----------
    checkpoint_path : str
        path to directory with checkpoint files saved by Tensorflow, to reload model
    networks : namedtuple
        where each field is the Config tuple for a neural network and the name
        of that field is the name of the class that represents the network.
    labels_mapping_path : str
        path to file that contains labels mapping, to convert output from consecutive
        digits back to labels used for audio segments (e.g. birdsong syllables)
    dir_to_predict : str
        path to directory where input files are located
    spect_scaler_path : str
        path to a saved SpectScaler object used to normalize spectrograms.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.
    """
    checkpoint_path = attr.ib(validator=[instance_of(str), is_a_directory])
    networks = attr.ib()
    labels_mapping_path = attr.ib(validator=[instance_of(str), is_a_file])
    dir_to_predict = attr.ib(validator=[instance_of(str), is_a_directory])
    spect_scaler_path = attr.ib(validator=optional([instance_of(str), is_a_file]), default=None)


def parse_predict_config(config):
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
        config_dict['checkpoint_path'] = config['PREDICT']['checkpoint_path']
    except NoOptionError:
        raise KeyError('must specify checkpoint_path in [PREDICT] section '
                       'of config.ini file')

    # load entry points within function, not at module level,
    # to avoid circular dependencies
    # (user would be unable to import networks in other packages
    # that subclass vak.network.AbstractVakNetwork
    # since the module in the other package would need to `import vak`)
    NETWORKS = network._load()
    NETWORK_NAMES = NETWORKS.keys()
    try:
        networks = [network_name for network_name in
                    config['PREDICT']['networks'].split(',')]
        for network_name in networks:
            if network_name not in NETWORK_NAMES:
                raise TypeError('Neural network {} not found when importing installed networks.'
                                .format(network))
        config_dict['networks'] = networks
    except NoOptionError:
        raise KeyError("'networks' option not found in [PREDICT] section of config.ini file. "
                       "Please add this option as a comma-separated list of neural network names, e.g.:\n"
                       "networks = TweetyNet, GRUnet, convnet")

    try:
        config_dict['labels_mapping_path'] = config['PREDICT']['labels_mapping_path']
    except NoOptionError:
        raise KeyError('must specify labels_mapping_path in [PREDICT] section '
                       'of config.ini file')

    try:
        config_dict['dir_to_predict'] = config['PREDICT']['dir_to_predict']
    except NoOptionError:
        raise KeyError('must specify dir_to_predict in [PREDICT] section '
                       'of config.ini file')

    if config.has_option('PREDICT','spect_scaler_path'):
        config_dict['spect_scaler_path'] = config['PREDICT']['spect_scaler_path']

    return PredictConfig(**config_dict)
