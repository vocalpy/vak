"""Class that represents ``[vak.learncurve]`` table in configuration file."""
from __future__ import annotations

from attrs import define, field
from attrs import converters, validators

from .eval import are_valid_post_tfm_kwargs, convert_post_tfm_kwargs
from .train import TrainConfig


REQUIRED_KEYS = (
    'dataset',
    'model',
    'root_results_dir'
)


@define
class LearncurveConfig(TrainConfig):
    """Class that represents ``[vak.learncurve]`` table in configuration file.

    Attributes
    ----------
    model : vak.config.ModelConfig
        The model to use: its name,
        and the parameters to configure it.
        Must be an instance of :class:`vak.config.ModelConfig`
    dataset : vak.config.DatasetConfig
        The dataset to use: the path to it,
        and optionally a path to a file representing splits,
        and the name, if it is a built-in dataset.
        Must be an instance of :class:`vak.config.DatasetConfig`.
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

    post_tfm_kwargs = field(
        validator=validators.optional(are_valid_post_tfm_kwargs),
        converter=converters.optional(convert_post_tfm_kwargs),
        default=None,
    )

    # we over-ride this method from TrainConfig mainly so the docstring is correct.
    # TODO: can we do this by just over-writing `__doc__` for the method on this class?
    @classmethod
    def from_config_dict(cls, config_dict: dict) -> "TrainConfig":
        """Return :class:`LearncurveConfig` instance from a :class:`dict`.

        The :class:`dict` passed in should be the one found
        by loading a valid configuration toml file with
        :func:`vak.config.parse.from_toml_path`,
        and then using key ``prep``,
        i.e., ``LearncurveConfig.from_config_dict(config_dict['train'])``."""
        for required_key in REQUIRED_KEYS:
            if required_key not in config_dict:
                raise KeyError(
                    "The `[vak.train]` table in a configuration file requires "
                    f"the option '{required_key}', but it was not found "
                    "when loading the configuration file into a Python dictionary. "
                    "Please check that the configuration file is formatted correctly."
                )
        config_dict['model'] = ModelConfig(**config_dict['model'])
        config_dict['dataset'] = DatasetConfig(**config_dict['dataset'])
        return cls(
            **config_dict
        )