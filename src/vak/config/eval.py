"""parses [EVAL] section of config"""
import attr
from attr import converters, validators
from attr.validators import instance_of

from .validators import is_valid_model_name
from .. import device
from ..converters import comma_separated_list, expanded_user_path


def convert_post_tfm_kwargs(post_tfm_kwargs: dict) -> dict:
    post_tfm_kwargs = dict(post_tfm_kwargs)

    if 'min_segment_dur' not in post_tfm_kwargs:
        # because there's no null in TOML,
        # users leave arg out of config then we set it to None
        post_tfm_kwargs['min_segment_dur'] = None
    else:
        post_tfm_kwargs['min_segment_dur'] = float(post_tfm_kwargs['min_segment_dur'])

    if 'majority_vote' not in post_tfm_kwargs:
        # set default for this one too
        post_tfm_kwargs['majority_vote'] = False
    else:
        post_tfm_kwargs['majority_vote'] = bool(post_tfm_kwargs['majority_vote'])

    return post_tfm_kwargs


def are_valid_post_tfm_kwargs(instance, attribute, value):
    """check if ``post_tfm_kwargs`` is valid"""
    if not isinstance(value, dict):
        raise TypeError(
            "'post_tfm_kwargs' should be declared in toml config as an inline table "
            f"that parses as a dict, but type was: {type(value)}. "
            "Please declare in a similar fashion: `{majority_vote = True, min_segment_dur = 0.02}`"
        )
    if any(
        [k not in {'majority_vote', 'min_segment_dur'} for k in value.keys()]
    ):
        invalid_kwargs = [k for k in value.keys()
                          if k not in {'majority_vote', 'min_segment_dur'}]
        raise ValueError(
            f"Invalid keyword argument name specified for 'post_tfm_kwargs': {invalid_kwargs}."
            "Valid names are: {'majority_vote', 'min_segment_dur'}"
        )
    if 'majority_vote' in value:
        if not isinstance(value['majority_vote'], bool):
            raise TypeError(
                "'post_tfm_kwargs' keyword argument 'majority_vote' "
                f"should be of type bool but was: {type(value['majority_vote'])}"
            )
    if 'min_segment_dur' in value:
        if value['min_segment_dur'] and not isinstance(value['min_segment_dur'], float):
            raise TypeError(
                "'post_tfm_kwargs' keyword argument 'min_segment_dur' type "
                f"should be float but was: {type(value['min_segment_dur'])}"
            )


@attr.s
class EvalConfig:
    """class that represents [EVAL] section of config.toml file

    Attributes
    ----------
    csv_path : str
        path to where dataset was saved as a csv.
    checkpoint_path : str
        path to directory with checkpoint files saved by Torch, to reload model
    output_dir : str
        Path to location where .csv files with evaluation metrics should be saved.
    labelmap_path : str
        path to 'labelmap.json' file.
    models : list
        of model names. e.g., 'models = TweetyNet, GRUNet, ConvNet'
    batch_size : int
        number of samples per batch presented to models during training.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader. Default is 2.
    device : str
        Device on which to work with model + data.
        Defaults to 'cuda' if torch.cuda.is_available is True.
    spect_scaler_path : str
        path to a saved SpectScaler object used to normalize spectrograms.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.
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
    # required, external files
    checkpoint_path = attr.ib(converter=expanded_user_path)
    labelmap_path = attr.ib(converter=expanded_user_path)
    output_dir = attr.ib(converter=expanded_user_path)

    # required, model / dataloader
    models = attr.ib(
        converter=comma_separated_list,
        validator=[instance_of(list), is_valid_model_name],
    )
    batch_size = attr.ib(converter=int, validator=instance_of(int))

    # csv_path is actually 'required' but we can't enforce that here because cli.prep looks at
    # what sections are defined to figure out where to add csv_path after it creates the csv
    csv_path = attr.ib(
        converter=converters.optional(expanded_user_path),
        default=None,
    )

    # optional, transform
    spect_scaler_path = attr.ib(
        converter=converters.optional(expanded_user_path),
        default=None,
    )

    post_tfm_kwargs = attr.ib(
        validator=validators.optional(are_valid_post_tfm_kwargs),
        converter=converters.optional(convert_post_tfm_kwargs),
        default={},  # empty dict so we can pass into transform with **kwargs expansion
    )

    # optional, data loader
    num_workers = attr.ib(validator=instance_of(int), default=2)
    device = attr.ib(validator=instance_of(str), default=device.get_default())
