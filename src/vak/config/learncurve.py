"""parses [LEARNCURVE] section of config"""
import attr
from attr import converters, validators

from .eval import are_valid_post_tfm_kwargs, convert_post_tfm_kwargs
from .train import TrainConfig


@attr.s
class LearncurveConfig(TrainConfig):
    """class that represents [LEARNCURVE] section of config.toml file

    Attributes
    ----------
    model : str
        Model name, e.g., ``model = "TweetyNet"``
    dataset_path : str
        Path to dataset, e.g., a csv file generated by running ``vak prep``.
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
    save_only_single_checkpoint_file : bool
        if True, save only one checkpoint file instead of separate files every time
        we save. Default is True.
    use_train_subsets_from_previous_run : bool
        if True, use training subsets saved in a previous run. Default is False.
        Requires setting previous_run_path option in config.toml file.
    post_tfm_kwargs : dict
        Keyword arguments to post-processing transform.
        If None, then no additional clean-up is applied
        when transforming labeled timebins to segments,
        the default behavior.
        The transform used is
        ``vak.transforms.frame_labels.ToSegmentsWithPostProcessing`.
        Valid keyword argument names are 'majority_vote'
        and 'min_segment_dur', and should be appropriate
        values for those arguments: Boolean for ``majority_vote``,
        a float value for ``min_segment_dur``.
        See the docstring of the transform for more details on
        these arguments and how they work.
    """
    post_tfm_kwargs = attr.ib(
        validator=validators.optional(are_valid_post_tfm_kwargs),
        converter=converters.optional(convert_post_tfm_kwargs),
        default=None,
    )
