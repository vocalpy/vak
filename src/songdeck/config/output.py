"""parses [OUTPUT] section of config"""
from collections import namedtuple


OutputConfig = namedtuple('OutputConfig', ['root_results_dir',
                                           ])


def parse_output_config(config):
    """parse [OUTPUT] section of config.ini file

    Parameters
    ----------
    config : ConfigParser
        containing config.ini file already loaded by parse function

    Returns
    -------
    output_config : namedtuple
        with fields:
            fft_size
            step_size
    """
    if config.has_option('OUTPUT', 'root_results_dir'):
        root_results_dir = config['OUTPUT']['root_results_dir']
    else:
        root_results_dir = None

    return OutputConfig(root_results_dir)
