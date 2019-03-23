"""parses [PREDICT] section of config"""
import os
from collections import namedtuple
from configparser import NoOptionError

from ..network import _load

PredictConfig = namedtuple('PredictConfig', ['checkpoint_path',
                                             'networks',
                                             'labels_mapping_path',
                                             'dir_to_predict',
                                             'spect_scaler_path'
                                            ])


def parse_predict_config(config):
    """parse [PREDICT] section of config.ini file

    Parameters
    ----------
    config : ConfigParser
        containing config.ini file already loaded by parse function

    Returns
    -------
    predict_config : namedtuple
        with fields:
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
    try:
        checkpoint_path = config['PREDICT']['checkpoint_path']
        if not os.path.isdir(checkpoint_path):
            raise NotADirectoryError("directory specified as saved model checkpoint, "
                                     "{}, was not found."
                                     .format(checkpoint_path))
    except NoOptionError:
        raise KeyError('must specify checkpoint_path in [PREDICT] section '
                       'of config.ini file')

    # load entry points within function, not at module level,
    # to avoid circular dependencies
    # (user would be unable to import networks in other packages
    # that subclass vak.network.AbstractVakNetwork
    # since the module in the other package would need to `import vak`)
    NETWORKS = _load()
    NETWORK_NAMES = [network_name.lower() for network_name in NETWORKS.keys()]
    try:
        networks = [network_name for network_name in
                    config['PREDICT']['networks'].split(',')]
        for network in networks:
            if network.lower() not in NETWORK_NAMES:
                raise TypeError('Neural network {} not found when importing installed networks.'
                                .format(network))
    except NoOptionError:
        raise KeyError("'networks' option not found in [TRAIN] section of config.ini file. "
                       "Please add this option as a comma-separated list of neural network names, e.g.:\n"
                       "networks = TweetyNet, GRUnet, convnet")

    try:
        labels_mapping_path = config['PREDICT']['labels_mapping_path']
        if not os.path.isfile(labels_mapping_path):
            raise NotADirectoryError("file specified as labels mapping, "
                                     "{}, was not found."
                                     .format(labels_mapping_path))
    except NoOptionError:
        raise KeyError('must specify checkpoint_path in [PREDICT] section '
                       'of config.ini file')

    try:
        dir_to_predict = config['PREDICT']['dir_to_predict']
        if not os.path.isdir(dir_to_predict):
            raise NotADirectoryError('directory {}, specified as '
                                     'dir_to_predict, was not found.'
                                     .format(dir_to_predict))
    except NoOptionError:
        raise KeyError('must specify dir_to_predict in [PREDICT] section '
                       'of config.ini file')

    try:
        spect_scaler_path = config['PREDICT']['spect_scaler_path']
        if not os.path.isfile(spect_scaler_path):
            raise NotADirectoryError("file specified as spect scaler, "
                                     "{}, was not found."
                                     .format(spect_scaler_path))
    except NoOptionError:
        spect_scaler_path = None

    return PredictConfig(checkpoint_path,
                         networks,
                         labels_mapping_path,
                         dir_to_predict,
                         spect_scaler_path)
