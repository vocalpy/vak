"""parses [PREDICT] section of config"""
import os
from collections import namedtuple
from configparser import NoOptionError

PredictConfig = namedtuple('OutputConfig', ['checkpoint_dir',
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
            checkpoint_dir : str
                directory with checkpoint files saved by Tensorflow, to reload model
            dir_to_predict : str
                directory with audio files for which predictions should be made
    """
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

    return PredictConfig(checkpoint_dir,
                         dir_to_predict)
