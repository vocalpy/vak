"""parses [TRAIN] section of config"""
import attr
from attr import converters, validators
from attr.validators import instance_of

from .validators import is_a_directory, is_a_file, is_valid_model_name
from .. import device
from ..converters import bool_from_str, comma_separated_list, expanded_user_path


@attr.s
class TrainConfig:
    """class that represents [TRAIN] section of config.toml file

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
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader.
    device : str
        Device on which to work with model + data.
        Defaults to 'cuda' if torch.cuda.is_available is True.
    shuffle: bool
        if True, shuffle training data before each epoch. Default is True.
    normalize_spectrograms : bool
        if True, use spect.utils.data.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    val_step : int
        Step on which to estimate accuracy using validation set.
        If val_step is n, then validation is carried out every time
        the global step / n is a whole number, i.e., when val_step modulo the global step is 0.
        Default is None, in which case no validation is done.
    ckpt_step : int
        Step on which to save to checkpoint file.
        If ckpt_step is n, then a checkpoint is saved every time
        the global step / n is a whole number, i.e., when ckpt_step modulo the global step is 0.
        Default is None, in which case checkpoint is only saved at the last epoch.
    patience : int
        number of validation steps to wait without performance on the
        validation set improving before stopping the training.
        Default is None, in which case training only stops after the specified number of epochs.
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
    device = attr.ib(validator=instance_of(str), default=device.get_default())
    shuffle = attr.ib(converter=bool_from_str, validator=instance_of(bool), default=True)

    val_step = attr.ib(converter=converters.optional(int),
                             validator=validators.optional(instance_of(int)), default=None)
    ckpt_step = attr.ib(converter=converters.optional(int),
                              validator=validators.optional(instance_of(int)), default=None)
    patience = attr.ib(converter=converters.optional(int),
                       validator=validators.optional(instance_of(int)), default=None)


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
                f"TRAIN section of the config.toml file: {toml_path}"
            )
    return TrainConfig(**train_section)
