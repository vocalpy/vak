"""parses [PREDICT] section of config"""
import os
from pathlib import Path

import attr
from attr import converters, validators
from attr.validators import instance_of

from .validators import is_a_directory, is_a_file, is_valid_model_name
from .. import device
from ..converters import comma_separated_list, expanded_user_path


@attr.s
class PredictConfig:
    """class that represents [PREDICT] section of config.toml file

    Attributes
    ----------
    csv_path : str
        path to where dataset was saved as a csv.
    checkpoint_path : str
        path to directory with checkpoint files saved by Torch, to reload model
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
    annot_csv_filename : str
        name of .csv file containing predicted annotations.
        Default is None, in which case the name of the dataset .csv
        is used, with '.annot.csv' appended to it.
    output_dir : str
        path to location where .csv containing predicted annotation
        should be saved. Defaults to current working directory.
    min_segment_dur : float
        minimum duration of segment, in seconds. If specified, then
        any segment with a duration less than min_segment_dur is
        removed from lbl_tb. Default is None, in which case no
        segments are removed.
    majority_vote : bool
        if True, transform segments containing multiple labels
        into segments with a single label by taking a "majority vote",
        i.e. assign all time bins in the segment the most frequently
        occurring label in the segment. This transform can only be
        applied if the labelmap contains an 'unlabeled' label,
        because unlabeled segments makes it possible to identify
        the labeled segments. Default is False.
    """
    # required, external files
    checkpoint_path = attr.ib(converter=expanded_user_path,
                              validator=is_a_file)
    labelmap_path = attr.ib(converter=expanded_user_path,
                            validator=is_a_file)

    # required, model / dataloader
    models = attr.ib(converter=comma_separated_list,
                     validator=[instance_of(list), is_valid_model_name])
    batch_size = attr.ib(converter=int, validator=instance_of(int))

    # csv_path is actually 'required' but we can't enforce that here because cli.prep looks at
    # what sections are defined to figure out where to add csv_path after it creates the csv
    csv_path = attr.ib(converter=converters.optional(expanded_user_path),
                       validator=validators.optional(is_a_file),
                       default=None
                       )

    # optional, transform
    spect_scaler_path = attr.ib(converter=converters.optional(expanded_user_path),
                                validator=validators.optional(is_a_file),
                                default=None)

    # optional, data loader
    num_workers = attr.ib(validator=instance_of(int), default=2)
    device = attr.ib(validator=instance_of(str), default=device.get_default())

    annot_csv_filename = attr.ib(validator=validators.optional(instance_of(str)), default=None)
    output_dir = attr.ib(converter=expanded_user_path, validator=is_a_directory, default=Path(os.getcwd()))
    min_segment_dur = attr.ib(validator=validators.optional(instance_of(float)), default=None)
    majority_vote = attr.ib(validator=instance_of(bool), default=True)


REQUIRED_PREDICT_OPTIONS = [
    'checkpoint_path',
    'labelmap_path',
    'models',
]


def parse_predict_config(config_toml, toml_path):
    """parse [PREDICT] section of config.toml file

    Parameters
    ----------
    config_toml : dict
        containing configuration file in TOML format, already loaded by parse function
    toml_path : Path
        path to a configuration file in TOML format (used for error messages)

    Returns
    -------
    predict_config : vak.config.predict.PredictConfig
        instance of PredictConfig class that represents [PREDICT] section
        of config.toml file
    """
    predict_section = dict(
        config_toml['PREDICT'].items()
    )

    for required_option in REQUIRED_PREDICT_OPTIONS:
        if required_option not in predict_section:
            raise KeyError(
                f"the '{required_option}' option is required but was not found in the "
                f"PREDICT section of the config.toml file: {toml_path}"
            )
    return PredictConfig(**predict_section)
