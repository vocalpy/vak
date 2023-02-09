"""parses [LEARNCURVE] section of config"""
import attr
from attr import converters, validators
from attr.validators import instance_of

from .eval import are_valid_post_tfm_kwargs, convert_post_tfm_kwargs
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
    post_tfm_kwargs : dict
        Keyword arguments to post-processing transform.
        If None, then no additional clean-up is applied
        when transforming labeled timebins to segments,
        the default behavior.
        The transform used is
        ``vak.transforms.labeled_timebins.ToSegmentsWithPostProcessing`.
        Valid keyword argument names are 'majority_vote'
        and 'min_segment_dur', and should be appropriate
        values for those arguments: Boolean for ``majority_vote``,
        a float value for ``min_segment_dur``.
        See the docstring of the transform for more details on
        these arguments and how they work.
    """
    train_set_durs = attr.ib(validator=instance_of(list), kw_only=True)
    num_replicates = attr.ib(validator=instance_of(int), kw_only=True)
    previous_run_path = attr.ib(
        converter=converters.optional(expanded_user_path),
        default=None,
    )

    post_tfm_kwargs = attr.ib(
        validator=validators.optional(are_valid_post_tfm_kwargs),
        converter=converters.optional(convert_post_tfm_kwargs),
        default={},  # empty dict so we can pass into transform with **kwargs expansion
    )
