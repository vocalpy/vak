"""parses [TRAIN] section of config"""
import attr
from attr import converters, validators
from attr.validators import instance_of

from .converters import bool_from_str, comma_separated_list, expanded_user_path
from .validators import is_a_directory, is_a_file, is_valid_model_name
from ..util.general import get_default_device


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
        directory in which results will be created.
        The vak.cli.train function will create
        a subdirectory in this directory each time it runs.
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
    num_epochs = attr.ib(converter=int, validator=instance_of(int))
    batch_size = attr.ib(converter=int, validator=instance_of(int))
    root_results_dir = attr.ib(converter=expanded_user_path, validator=is_a_directory)

    # optional
    # csv_path is actually 'required' but we can't enforce that here because cli.prep looks at
    # what sections are defined to figure out where to add csv_path after it creates the csv
    csv_path = attr.ib(converter=converters.optional(expanded_user_path),
                       validator=validators.optional(is_a_file),
                       default=None
                       )

    results_dirname = attr.ib(converter=converters.optional(expanded_user_path),
                              validator=validators.optional(is_a_directory), default=None)

    normalize_spectrograms = attr.ib(converter=bool_from_str,
                                     validator=validators.optional(instance_of(bool)), default=False)

    num_workers = attr.ib(validator=instance_of(int), default=2)
    device = attr.ib(validator=instance_of(str), default=get_default_device())
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
    'root_results_dir',
]


def parse_train_config(config_toml, toml_path):
    """parse [TRAIN] section of config.toml file

    Parameters
    ----------
    config_toml : dict
        containing configuration file in TOML format, already loaded by parse function
    toml_path : Path
        path to a configuration file in TOML format (used for error messages)

    Returns
    -------
    train_config : vak.config.train.TrainConfig
        instance of TrainConfig class
    """
    train_section = config_toml['TRAIN']
    train_section = dict(train_section.items())
    for required_option in REQUIRED_TRAIN_OPTIONS:
        if required_option not in train_section:
            raise KeyError(
                f"the '{required_option}' option is required but was not found in the "
                f"TRAIN section of the config.ini file: {toml_path}"
            )
    return TrainConfig(**train_section)
