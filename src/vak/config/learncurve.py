"""parses [LEARNCURVE] section of config"""
import attr
from attr import converters, validators
from attr.validators import instance_of

from .validators import is_a_directory
from .train import TrainConfig
from ..converters import expanded_user_path


@attr.s
class LearncurveConfig(TrainConfig):
    """class that represents [LEARNCURVE] section of config.toml file

    Attributes
    ----------
    models : list
        of model names. e.g., 'models = TweetyNet, GRUNet, ConvNet'
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
    ckpt_step : int
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
        Requires setting previous_run_path option in config.toml file.
    previous_run_path : str
        path to results directory from a previous run.
        Used for training if use_train_subsets_from_previous_run is True.
    """
    train_set_durs = attr.ib(validator=instance_of(list), kw_only=True)
    num_replicates = attr.ib(validator=instance_of(int), kw_only=True)
    previous_run_path = attr.ib(converter=converters.optional(expanded_user_path),
                                validator=validators.optional(is_a_directory), default=None)


REQUIRED_LEARNCURVE_OPTIONS = [
    'models',
    'root_results_dir',
    'train_set_durs',
    'num_replicates',
]


def parse_learncurve_config(config_toml, toml_path):
    """parse [LEARNCURVE] section of config.toml file

    Parameters
    ----------
    config_toml : dict
        containing configuration file in TOML format, already loaded by parse function
    toml_path : Path
        path to a configuration file in TOML format (used for error messages)

    Returns
    -------
    learncurve_config : vak.config.learncurve.LearncurveConfig
        instance of LearncurveConfig class
    """
    learncurve_section = config_toml['LEARNCURVE']
    learncurve_section = dict(learncurve_section.items())
    for required_option in REQUIRED_LEARNCURVE_OPTIONS:
        if required_option not in learncurve_section:
            raise KeyError(
                f"the '{required_option}' option is required but was not found in the "
                f"LEARNCURVE section of the config.toml file: {toml_path}"
            )

    return LearncurveConfig(**learncurve_section)
