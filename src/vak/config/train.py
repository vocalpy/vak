"""parses [TRAIN] section of config"""
from configparser import NoOptionError

import attr
from attr.validators import instance_of, optional

from .validators import is_a_directory, is_a_file
from .. import network


@attr.s
class TrainConfig:
    """class that represents [TRAIN] section of config.ini file

    Attributes
    ----------
    networks : namedtuple
        where each field is the Config tuple for a neural network and the name
        of that field is the name of the class that represents the network.
    train_data_dict_path : str
        path to training data
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    normalize_spectrograms : bool
        if True, use spect.utils.data.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    val_data_dict_path : str
        path to validation data. Default is None, in which case accuracy is not measured
        on a validation set during training.
    val_error_step : int
        step/epoch at which to estimate accuracy using validation set.
        Default is None, in which case no validation is done.
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
    save_transformed_data : bool
        if True, save transformed data (i.e. scaled, reshaped). The data can then
        be used on a subsequent run of learncurve (e.g. if you want to compare results
        from different hyperparameters across the exact same training set).
        Also useful if you need to check what the data looks like when fed to networks.
    """
    # required for both train and learncurve
    networks = attr.ib(validator=instance_of(list))
    train_data_dict_path = attr.ib(validator=[instance_of(str), is_a_file])
    num_epochs = attr.ib(validator=instance_of(int))

    # used for both train and learncurve, but optional
    normalize_spectrograms = attr.ib(validator=optional(instance_of(bool)), default=False)
    val_data_dict_path = attr.ib(validator=optional([instance_of(str), is_a_file]), default=None)
    test_data_dict_path = attr.ib(validator=optional([instance_of(str), is_a_file]), default=None)
    val_error_step = attr.ib(validator=optional(instance_of(int)), default=None)
    checkpoint_step = attr.ib(validator=optional(instance_of(int)), default=None)
    patience = attr.ib(validator=optional(instance_of(int)), default=None)

    # used for learncurve, not train
    train_set_durs = attr.ib(validator=optional(instance_of(list)), default=None)
    num_replicates = attr.ib(validator=optional(instance_of(int)), default=None)

    # things most users probably won't care about
    save_only_single_checkpoint_file = attr.ib(validator=instance_of(bool), default=True)
    use_train_subsets_from_previous_run = attr.ib(validator=instance_of(bool), default=False)
    previous_run_path = attr.ib(validator=optional([instance_of(str), is_a_directory]), default=None)
    save_transformed_data = attr.ib(validator=instance_of(bool), default=False)


def parse_train_config(config, config_file):
    """parse [TRAIN] section of config.ini file

    Parameters
    ----------
    config : ConfigParser
        containing config.ini file already loaded by parse function

    Returns
    -------
    train_config : vak.config.train.TrainConfig
        instance of TrainConfig class
    """
    config_dict = {}
    # load entry points within function, not at module level,
    # to avoid circular dependencies
    # (user would be unable to import networks in other packages
    # that subclass vak.network.AbstractVakNetwork
    # since the module in the other package would need to `import vak`)
    NETWORKS = network._load()
    NETWORK_NAMES = NETWORKS.keys()
    try:
        networks = [network_name for network_name in
                    config['TRAIN']['networks'].split(',')]
        for network_name in networks:
            if network_name not in NETWORK_NAMES:
                raise TypeError('Neural network {} not found when importing installed networks.'
                                .format(network))
        config_dict['networks'] = networks
    except NoOptionError:
        raise KeyError("'networks' option not found in [TRAIN] section of config.ini file. "
                       "Please add this option as a comma-separated list of neural network names, e.g.:\n"
                       "networks = TweetyNet, GRUnet, convnet")

    try:
        config_dict['train_data_dict_path'] = config['TRAIN']['train_data_path']
    except NoOptionError:
        raise KeyError("'train_data_path' option not found in [TRAIN] section of config.ini file. "
                       "Please add this option.")

    if config.has_option('TRAIN', 'train_set_durs'):
        config_dict['train_set_durs'] = [int(element)
                                         for element in
                                         config['TRAIN']['train_set_durs'].split(',')]

    if config.has_option('TRAIN', 'replicates'):
        config_dict['num_replicates'] = int(config['TRAIN']['replicates'])

    if config.has_option('TRAIN', 'val_data_path'):
        config_dict['val_data_dict_path'] = config['TRAIN']['val_data_path']

    if config.has_option('TRAIN', 'test_data_path'):
        config_dict['test_data_dict_path'] = config['TRAIN']['test_data_path']

    if config.has_option('TRAIN', 'val_error_step'):
        config_dict['val_error_step'] = int(config['TRAIN']['val_error_step'])

    if config.has_option('TRAIN', 'checkpoint_step'):
        config_dict['checkpoint_step'] = int(config['TRAIN']['checkpoint_step'])

    if config.has_option('TRAIN', 'save_only_single_checkpoint_file'):
        config_dict['save_only_single_checkpoint_file'] = config.getboolean(
            'TRAIN','save_only_single_checkpoint_file'
        )

    if config.has_option('TRAIN', 'num_epochs'):
        config_dict['num_epochs'] = int(config['TRAIN']['num_epochs'])

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
        config_dict['patience'] = patience

    if config.has_option('TRAIN', 'normalize_spectrograms'):
        config_dict['normalize_spectrograms'] = config.getboolean(
            'TRAIN', 'normalize_spectrograms'
        )

    if config.has_option('TRAIN', 'use_train_subsets_from_previous_run'):
        config_dict['use_train_subsets_from_previous_run'] = config.getboolean(
            'TRAIN', 'use_train_subsets_from_previous_run')
        if config_dict['use_train_subsets_from_previous_run']:
            try:
                config_dict['previous_run_path'] = config['TRAIN']['previous_run_path']
            except KeyError:
                raise KeyError('In config.file {}, '
                               'use_train_subsets_from_previous_run = Yes, but '
                               'no previous_run_path option was found.'
                               'Please add previous_run_path to config file.'
                               .format(config_file))
        else:
            if config.has_option('TRAIN', 'previous_run_path'):
                raise ValueError('In config.file {}, '
                                 'use_train_subsets_from_previous_run = No, but '
                                 'previous_run_path option was specified as {}.\n'
                                 'Please fix argument or remove/comment out '
                                 'previous_run_path.'
                                 .format(config_file,
                                         config['TRAIN']['previous_run_path'])
                                 )

    if config.has_option('TRAIN', 'save_transformed_data'):
        config_dict['save_transformed_data'] = config.getboolean(
            'TRAIN', 'save_transformed_data')

    return TrainConfig(**config_dict)
