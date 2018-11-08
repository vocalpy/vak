"""parses [OUTPUT] section of config"""
import os
from collections import namedtuple
from configparser import NoOptionError

OutputConfig = namedtuple('OutputConfig', ['root_results_dir',
                                           'results_dirname',
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
            root_results_dir : str
                directory in which results *will* be created.
                The songdeck.cli.learcurve function will create
                a subdirectory in this directory each time it runs.
            results_dirname : str
                name of subdirectory created by songdeck.cli.learncurve.
                This option is added programatically by that function
                when it runs but can be changed (e.g. to run
                songdeck.cli.summary on previous outputs of learncurve.)
    """
    try:
        root_results_dir = config['OUTPUT']['root_results_dir']
        if not os.path.isdir(root_results_dir):
            raise FileNotFoundError('directory {}, specified as '
                                    'root_results_dir, was not found.'
                                    .format(root_results_dir))
    except NoOptionError:
        raise KeyError('must specify root_results_dir in [OUTPUT] section '
                       'of config.ini file')

    try:
        results_dirname = config['OUTPUT']['results_dir_made_by_main_script']
        if not os.path.isdir(results_dirname):
            raise FileNotFoundError('directory {}, specified as '
                                    'results_dir_made_by_main_script, is not found.'
                                    .format(results_dirname))
    except NoOptionError:
        raise KeyError('must specify results_dir_made_by_main_script '
                            'in [OUTPUT] section of config.ini file')

    return OutputConfig(root_results_dir,
                        results_dirname)
