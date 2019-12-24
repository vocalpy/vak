"""parses [TRAIN] section of config"""
import os
from configparser import NoOptionError

import attr
from attr.validators import instance_of, optional

from .validators import is_a_directory, is_a_file
from .. import models


@attr.s
class TrainConfig:
    """class that represents [TRAIN] section of config.ini file

    Attributes
    ----------
    models : list
        comma-separated list of model names.
        e.g., 'models = TweetyNet, GRUNet, ConvNet'
    csv_path : str
        path to where dataset was saved as a csv.
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    root_results_dir : str
        directory in which results *will* be created.
        The vak.cli.train function will create
        a subdirectory in this directory each time it runs.
    results_dirname : str
        name of subdirectory created by vak.cli.train.
        This option is added programatically by that function
        when it runs.
    normalize_spectrograms : bool
        if True, use spect.utils.data.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    val_error_step : int
        step/epoch at which to estimate accuracy using validation set.
        Default is None, in which case no validation is done.
    checkpoint_step : int
        step/epoch at which to save to checkpoint file.
        Default is None, in which case checkpoint is only saved at the last epoch.
    patience : int
        number of epochs to wait without the error dropping before stopping the
        training. Default is None, in which case training continues for num_epochs
    save_only_single_checkpoint_file : bool
        if True, save only one checkpoint file instead of separate files every time
        we save. Default is True.
    """
    # required
    models = attr.ib(validator=instance_of(list))
    csv_path = attr.ib(validator=[instance_of(str), is_a_file])
    num_epochs = attr.ib(validator=instance_of(int))
    root_results_dir = attr.ib(validator=is_a_directory)
    results_dirname = attr.ib(validator=optional(is_a_directory), default=None)

    # optional
    normalize_spectrograms = attr.ib(validator=optional(instance_of(bool)), default=False)
    val_error_step = attr.ib(validator=optional(instance_of(int)), default=None)
    checkpoint_step = attr.ib(validator=optional(instance_of(int)), default=None)
    patience = attr.ib(validator=optional(instance_of(int)), default=None)
    save_only_single_checkpoint_file = attr.ib(validator=instance_of(bool), default=True)


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
    MODEL_NAMES = [model_name for model_name, model_builder in models.find()]
    try:
        model_names = [model_name
                       for model_name in config['TRAIN']['models'].split(',')]
    except NoOptionError:
        raise KeyError("'models' option not found in [TRAIN] section of config.ini file. "
                       "Please add this option as a comma-separated list of model names, e.g.:\n"
                       "models = TweetyNet, GRUnet, convnet")
    for model_name in model_names:
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Model {model_name} not found when importing installed models.'
            )
    config_dict['models'] = model_names

    try:
        config_dict['csv_path'] = os.path.expanduser(config['TRAIN']['csv_path'])
    except NoOptionError:
        raise KeyError("'csv_path' option not found in [TRAIN] section of config.ini file. "
                       "Please add this option.")

    try:
        root_results_dir = config['TRAIN']['root_results_dir']
        config_dict['root_results_dir'] = os.path.expanduser(root_results_dir)
    except NoOptionError:
        raise KeyError('must specify root_results_dir in [TRAIN] section '
                       'of config.ini file')

    if config.has_option('TRAIN', 'results_dir_made_by_main_script'):
        # don't check whether it exists because it may not yet,
        # depending on which function we are calling.
        # So it's up to calling function to check for existence of directory
        results_dirname = config['TRAIN']['results_dir_made_by_main_script']
        config_dict['results_dirname'] = os.path.expanduser(results_dirname)
    else:
        config_dict['results_dirname'] = None

    if config.has_option('TRAIN', 'val_error_step'):
        config_dict['val_error_step'] = int(config['TRAIN']['val_error_step'])

    if config.has_option('TRAIN', 'checkpoint_step'):
        config_dict['checkpoint_step'] = int(config['TRAIN']['checkpoint_step'])

    if config.has_option('TRAIN', 'save_only_single_checkpoint_file'):
        config_dict['save_only_single_checkpoint_file'] = config.getboolean(
            'TRAIN', 'save_only_single_checkpoint_file'
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

    return TrainConfig(**config_dict)
