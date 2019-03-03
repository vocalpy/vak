"""parses [TRAIN] section of config"""
from collections import namedtuple
from configparser import NoOptionError
from ..network import _load


TrainConfig = namedtuple('TrainConfig', [
    'networks',
    'train_data_dict_path',
    'train_set_durs',
    'num_replicates',
    'val_data_dict_path',
    'test_data_dict_path',
    'val_error_step',
    'checkpoint_step',
    'save_only_single_checkpoint_file',
    'num_epochs',
    'patience',
    'normalize_spectrograms',
    'use_train_subsets_from_previous_run',
    'previous_run_path',
    'save_transformed_data'
])


def parse_train_config(config, config_file):
    """parse [TRAIN] section of config.ini file

    Parameters
    ----------
    config : ConfigParser
        containing config.ini file already loaded by parse function

    Returns
    -------
    train_config : namedtuple
        with fields:
            networks :
            train_data_dict_path :
            train_set_durs :
            num_replicates :
            val_data_dict_path :
            val_error_step :
            checkpoint_step :
            save_only_single_checkpoint_file :
            num_epochs :
            patience :
            use_train_subsets_from_previous_run :
            previous_run_path :
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
                    config['TRAIN']['networks'].split(',')]
        for network in networks:
            if network.lower() not in NETWORK_NAMES:
                raise TypeError('Neural network {} not found when importing installed networks.'
                                .format(network))
    except NoOptionError:
        raise KeyError("'networks' option not found in [TRAIN] section of config.ini file. "
                       "Please add this option as a comma-separated list of neural network names, e.g.:\n"
                       "networks = TweetyNet, GRUnet, convnet")

    try:
        train_data_dict_path = config['TRAIN']['train_data_path']
    except NoOptionError:
        raise KeyError("'train_data_path' option not found in [TRAIN] section of config.ini file. "
                       "Please add this option.")

    if config.has_option('TRAIN', 'train_set_durs'):
        train_set_durs = [int(element)
                          for element in
                          config['TRAIN']['train_set_durs'].split(',')]
    else:
        # set to None when training on entire dataset
        train_set_durs = None

    if config.has_option('TRAIN', 'replicates'):
        num_replicates = int(config['TRAIN']['replicates'])
    else:
        # set to None when training on entire dataset
        num_replicates = None

    if config.has_option('TRAIN', 'val_data_path'):
        val_data_dict_path = config['TRAIN']['val_data_path']
    else:
        val_data_dict_path = None

    if config.has_option('TRAIN', 'test_data_path'):
        test_data_dict_path = config['TRAIN']['test_data_path']
    else:
        test_data_dict_path = None

    if config.has_option('TRAIN', 'val_error_step'):
        val_error_step = int(config['TRAIN']['val_error_step'])
    else:
        val_error_step = None

    if config.has_option('TRAIN', 'checkpoint_step'):
        checkpoint_step = int(config['TRAIN']['checkpoint_step'])
    else:
        checkpoint_step = None

    if config.has_option('TRAIN', 'save_only_single_checkpoint_file'):
        save_only_single_checkpoint_file = config.getboolean('TRAIN',
                                                             'save_only_single_checkpoint_file')
    else:
        save_only_single_checkpoint_file = True

    if config.has_option('TRAIN', 'num_epochs'):
        num_epochs = int(config['TRAIN']['num_epochs'])
    else:
        num_epochs = 18000

    if config.has_option('TRAIN', 'patience'):
        patience = config['TRAIN']['patience']
        try:
            patience = int(patience)
        except ValueError:
            if patience == 'None':
                patience = None
            else:
                raise TypeError('patience must be an int or None, but'
                                'is {} and parsed as type {}'
                                .format(patience, type(patience)))
    else:
        patience = None

    if config.has_option('TRAIN', 'normalize_spectrograms'):
        normalize_spectrograms = config.getboolean('TRAIN',
                                                   'normalize_spectrograms')
    else:
        normalize_spectrograms = False

    if config.has_option('TRAIN', 'use_train_subsets_from_previous_run'):
        use_train_subsets_from_previous_run = config.getboolean(
            'TRAIN', 'use_train_subsets_from_previous_run')
        if use_train_subsets_from_previous_run:
            try:
                previous_run_path = config['TRAIN']['previous_run_path']
            except KeyError:
                raise KeyError('In config.file {}, '
                               'use_train_subsets_from_previous_run = Yes, but '
                               'no previous_run_path option was found.'
                               'Please add previous_run_path to config file.'
                               .format(config_file))
        else:
            previous_run_path = None
    else:
        use_train_subsets_from_previous_run = False
        if config.has_option('TRAIN', 'previous_run_path'):
            raise ValueError('In config.file {}, '
                             'use_train_subsets_from_previous_run = No, but '
                             'previous_run_path option was specified as {}.\n'
                             'Please fix argument or remove/comment out '
                             'previous_run_path.'
                             .format(config_file,
                                     config['TRAIN']['previous_run_path'])
                             )
        else:
            previous_run_path = None

    if config.has_option('TRAIN', 'save_transformed_data'):
        save_transformed_data = config.getboolean(
            'TRAIN', 'save_transformed_data')
    else:
        save_transformed_data = False

    return TrainConfig(networks=networks,
                       train_data_dict_path=train_data_dict_path,
                       train_set_durs=train_set_durs,
                       num_replicates=num_replicates,
                       val_data_dict_path=val_data_dict_path,
                       test_data_dict_path=test_data_dict_path,
                       val_error_step=val_error_step,
                       checkpoint_step=checkpoint_step,
                       save_only_single_checkpoint_file=save_only_single_checkpoint_file,
                       num_epochs=num_epochs,
                       patience=patience,
                       normalize_spectrograms=normalize_spectrograms,
                       use_train_subsets_from_previous_run=use_train_subsets_from_previous_run,
                       previous_run_path=previous_run_path,
                       save_transformed_data=save_transformed_data)
