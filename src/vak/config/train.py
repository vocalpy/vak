"""parses [TRAIN] section of config"""
from configparser import NoOptionError

import attr
from attr import converters, validators
from attr.validators import instance_of

from .converters import bool_from_str, comma_separated_list, expanded_user_path
from .validators import is_a_directory, is_a_file, is_valid_model_name


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
    batch_size : int
        number of samples per batch presented to models during training.
    root_results_dir : str
        directory in which results *will* be created.
        The vak.cli.train function will create
        a subdirectory in this directory each time it runs.
    results_dirname : str
        name of subdirectory created by vak.cli.train.
        This option is added programatically by that function
        when it runs.
    optimizer : str
        name of numerical optimizer used to fit model, e.g. 'SGD' for stochastic gradient descent.
    learning rate : float
        value used
    loss : str
        name of loss function used to fit model, e.g. 'categorical_crossentropy'
    metrics : str, list, dict
        metrics evaluated by model during training and testing, e.g. ['accuracy', 'mse']
    shuffle: bool
        if True, shuffle training data before each epoch. Default is True.
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
    models = attr.ib(converter=comma_separated_list,
                     validator=[instance_of(list), is_valid_model_name])
    csv_path = attr.ib(converter=expanded_user_path, validator=is_a_file)
    num_epochs = attr.ib(converter=int, validator=instance_of(int))
    batch_size = attr.ib(converter=int, validator=instance_of(int))
    root_results_dir = attr.ib(converter=expanded_user_path, validator=is_a_directory)
    optimizer = attr.ib(validator=instance_of(str))
    learning_rate = attr.ib(converter=float, validator=instance_of(float))
    loss = attr.ib(validator=instance_of(str))
    metrics = attr.ib(converter=comma_separated_list)

    # optional
    results_dirname = attr.ib(converter=converters.optional(expanded_user_path),
                              validator=validators.optional(is_a_directory), default=None)
    normalize_spectrograms = attr.ib(converter=bool_from_str,
                                     validator=validators.optional(instance_of(bool)), default=False)
    shuffle = attr.ib(converter=bool_from_str, validator=instance_of(bool), default=True)
    val_error_step = attr.ib(converter=converters.optional(int),
                             validator=validators.optional(instance_of(int)), default=None)
    checkpoint_step = attr.ib(converter=converters.optional(int),
                              validator=validators.optional(instance_of(int)), default=None)
    patience = attr.ib(converter=converters.optional(int),
                       validator=validators.optional(instance_of(int)), default=None)
    save_only_single_checkpoint_file = attr.ib(converter=bool_from_str,
                                               validator=instance_of(bool), default=True)


REQUIRED_TRAIN_OPTIONS = [
    'models',
    'csv_path',
    'root_results_dir',
]


def parse_train_config(config, config_path):
    """parse [TRAIN] section of config.ini file

    Parameters
    ----------
    config : ConfigParser
        containing config.ini file already loaded by parse function
    config_path : str
        path to config.ini file (used for error messages)

    Returns
    -------
    train_config : vak.config.train.TrainConfig
        instance of TrainConfig class
    """
    train_section = config['TRAIN']
    train_section = dict(train_section.items())
    for required_option in REQUIRED_TRAIN_OPTIONS:
        if required_option not in train_section:
            raise NoOptionError(
                f"the '{required_option}' option is required but was not found in the "
                f"TRAIN section of the config.ini file: {config_path}"
            )
    return TrainConfig(**train_section)
