"""parses [PREDICT] section of config"""
import os
from collections import namedtuple
from configparser import NoOptionError

from ..network import _load

PredictConfig = namedtuple('OutputConfig', ['networks',
                                            'checkpoint_dir',
                                            'dir_to_predict',
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
            networks : str
                Name of network which was trained and which should be used to make
                predictions. This must match the original network that was trained
                for the checkpoint file used and is required so vak knows which
                type of network to load before loading the parameters saved in the
                checkpoint file.
            checkpoint_dir : str
                directory with checkpoint files saved by Tensorflow, to reload model
            dir_to_predict : str
                directory with audio files for which predictions should be made
    """
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
        checkpoint_dir = config['PREDICT']['checkpoint_dir']
        if not os.path.isdir(checkpoint_dir):
            raise NotADirectoryError('directory {}, specified as '
                                     'checkpoint_dir, was not found.'
                                     .format(checkpoint_dir))
    except NoOptionError:
        raise KeyError('must specify checkpoint_dir in [PREDICT] section '
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

    return PredictConfig(networks,
                         checkpoint_dir,
                         dir_to_predict)
