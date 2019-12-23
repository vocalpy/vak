"""parses [LEARNCURVE] section of config"""
import os
from configparser import NoOptionError

import attr
from attr.validators import instance_of, optional

from .train import TrainConfig
from .validators import is_a_directory, is_a_file
from .. import models


@attr.s
class LearncurveConfig(TrainConfig):
    """class that represents [LEARNCURVE] section of config.ini file

    Attributes
    ----------
    networks : namedtuple
        where each field is the Config tuple for a neural network and the name
        of that field is the name of the class that represents the network.
    csv_path : str
        path to where dataset was saved as a csv.
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    normalize_spectrograms : bool
        if True, use spect.utils.data.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    checkpoint_step : int
        step/epoch at which to save to checkpoint file.
        Default is None, in which case checkpoint is only saved at the last epoch.
    patience : int
        number of epochs to wait without the error dropping before stopping the
        training. Default is None, in which case training continues for num_epochs
    train_set_durs : list
        of int, durations in seconds of subsets taken from training data
        to create a learning curve, e.g. [5, 10, 15, 20]. Default is None
        (when training a single model on all available training data).
    num_replicates : int
        number of times to replicate training for each training set duration
        to better estimate mean accuracy for a training set of that size.
        Each replicate uses a different randomly drawn subset of the training
        data (but of the same duration).
    save_only_single_checkpoint_file : bool
        if True, save only one checkpoint file instead of separate files every time
        we save. Default is True.
    use_train_subsets_from_previous_run : bool
        if True, use training subsets saved in a previous run. Default is False.
        Requires setting previous_run_path option in config.ini file.
    previous_run_path : str
        path to results directory from a previous run.
        Used for training if use_train_subsets_from_previous_run is True.
    """
    train_set_durs = attr.ib(validator=instance_of(list), kw_only=True)
    num_replicates = attr.ib(validator=instance_of(int), kw_only=True)


def parse_learncurve_config(config, config_file):
    """parse [LEARNCURVE] section of config.ini file

    Parameters
    ----------
    config : ConfigParser
        containing config.ini file already loaded by parse function

    Returns
    -------
    learncurve_config : vak.config.train.LearncurveConfig
        instance of LearncurveConfig class
    """
    config_dict = {}
    # load entry points within function, not at module level,
    # to avoid circular dependencies
    # (user would be unable to import networks in other packages
    # that subclass vak.network.AbstractVakNetwork
    # since the module in the other package would need to `import vak`)
    NETWORKS = models._load()
    NETWORK_NAMES = NETWORKS.keys()
    try:
        networks = [network_name for network_name in
                    config['LEARNCURVE']['networks'].split(',')]
        for network_name in networks:
            if network_name not in NETWORK_NAMES:
                raise TypeError(
                    f'Neural network {network_name} not found when importing installed networks.'
                )
        config_dict['networks'] = networks
    except NoOptionError:
        raise KeyError("'networks' option not found in [LEARNCURVE] section of config.ini file. "
                       "Please add this option as a comma-separated list of neural network names, e.g.:\n"
                       "networks = TweetyNet, GRUnet, convnet")

    try:
        config_dict['train_vds_path'] = os.path.expanduser(config['LEARNCURVE']['train_vds_path'])
    except NoOptionError:
        raise KeyError("'train_vds_path' option not found in [LEARNCURVE] section of config.ini file. "
                       "Please add this option.")

    try:
        root_results_dir = config['LEARNCURVE']['root_results_dir']
        config_dict['root_results_dir'] = os.path.expanduser(root_results_dir)
    except NoOptionError:
        raise KeyError('must specify root_results_dir in [LEARNCURVE] section '
                       'of config.ini file')

    if config.has_option('LEARNCURVE', 'results_dir_made_by_main_script'):
        # don't check whether it exists because it may not yet,
        # depending on which function we are calling.
        # So it's up to calling function to check for existence of directory
        results_dirname = config['LEARNCURVE']['results_dir_made_by_main_script']
        config_dict['results_dirname'] = os.path.expanduser(results_dirname)
    else:
        config_dict['results_dirname'] = None

    if config.has_option('LEARNCURVE', 'train_set_durs'):
        config_dict['train_set_durs'] = [int(element)
                                         for element in
                                         config['LEARNCURVE']['train_set_durs'].split(',')]

    if config.has_option('LEARNCURVE', 'replicates'):
        config_dict['num_replicates'] = int(config['LEARNCURVE']['replicates'])

    if config.has_option('LEARNCURVE', 'val_vds_path'):
        config_dict['val_vds_path'] = os.path.expanduser(config['LEARNCURVE']['val_vds_path'])

    if config.has_option('LEARNCURVE', 'test_vds_path'):
        config_dict['test_vds_path'] = os.path.expanduser(config['LEARNCURVE']['test_vds_path'])

    if config.has_option('LEARNCURVE', 'val_error_step'):
        config_dict['val_error_step'] = int(config['LEARNCURVE']['val_error_step'])

    if config.has_option('LEARNCURVE', 'checkpoint_step'):
        config_dict['checkpoint_step'] = int(config['LEARNCURVE']['checkpoint_step'])

    if config.has_option('LEARNCURVE', 'save_only_single_checkpoint_file'):
        config_dict['save_only_single_checkpoint_file'] = config.getboolean(
            'LEARNCURVE', 'save_only_single_checkpoint_file'
        )

    if config.has_option('LEARNCURVE', 'num_epochs'):
        config_dict['num_epochs'] = int(config['LEARNCURVE']['num_epochs'])

    if config.has_option('LEARNCURVE', 'patience'):
        patience = config['LEARNCURVE']['patience']
        try:
            patience = int(patience)
        except ValueError:
            if patience == 'None':
                patience = None
            else:
                raise TypeError('patience must be an int or None, but'
                                'is {} and parsed as type {}'
                                .format(patience, type(patience)))
        config_dict['patience'] = patience

    if config.has_option('LEARNCURVE', 'normalize_spectrograms'):
        config_dict['normalize_spectrograms'] = config.getboolean(
            'LEARNCURVE', 'normalize_spectrograms'
        )

    if config.has_option('LEARNCURVE', 'use_train_subsets_from_previous_run'):
        config_dict['use_train_subsets_from_previous_run'] = config.getboolean(
            'LEARNCURVE', 'use_train_subsets_from_previous_run')
        if config_dict['use_train_subsets_from_previous_run']:
            try:
                config_dict['previous_run_path'] = os.path.expanduser(config['LEARNCURVE']['previous_run_path'])
            except KeyError:
                raise KeyError('In config.file {}, '
                               'use_train_subsets_from_previous_run = Yes, but '
                               'no previous_run_path option was found.'
                               'Please add previous_run_path to config file.'
                               .format(config_file))
        else:
            if config.has_option('LEARNCURVE', 'previous_run_path'):
                raise ValueError('In config.file {}, '
                                 'use_train_subsets_from_previous_run = No, but '
                                 'previous_run_path option was specified as {}.\n'
                                 'Please fix argument or remove/comment out '
                                 'previous_run_path.'
                                 .format(config_file,
                                         config['LEARNCURVE']['previous_run_path'])
                                 )

    if config.has_option('LEARNCURVE', 'save_transformed_data'):
        config_dict['save_transformed_data'] = config.getboolean(
            'LEARNCURVE', 'save_transformed_data')

    return LearncurveConfig(**config_dict)
