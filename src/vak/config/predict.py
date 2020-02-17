"""parses [PREDICT] section of config"""
from configparser import NoOptionError

import attr
from attr import converters, validators
from attr.validators import instance_of

from .converters import comma_separated_list, expanded_user_path
from .validators import is_a_file, is_valid_model_name, is_annot_format
from ..util.general import get_default_device


@attr.s
class PredictConfig:
    """class that represents [PREDICT] section of config.ini file

    Attributes
    ----------
    csv_path : str
        path to where dataset was saved as a csv.
    checkpoint_path : str
        path to directory with checkpoint files saved by Torch, to reload model
    labelmap_path : str
        path to 'labelmap.json' file.
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid.
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
    """
    # required, external files
    checkpoint_path = attr.ib(converter=expanded_user_path,
                              validator=is_a_file)
    labelmap_path = attr.ib(converter=expanded_user_path,
                            validator=is_a_file)

    # required, for annotation
    annot_format = attr.ib(validator=is_annot_format)

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

    # optional
    to_format_kwargs = attr.ib(validator=validators.optional(instance_of(dict)),
                               default=None)

    # optional, transform
    spect_scaler_path = attr.ib(converter=converters.optional(expanded_user_path),
                                validator=validators.optional(is_a_file),
                                default=None)

    # optional, data loader
    num_workers = attr.ib(validator=instance_of(int), default=2)
    device = attr.ib(validator=instance_of(str), default=get_default_device())


REQUIRED_PREDICT_OPTIONS = [
    'checkpoint_path',
    'labelmap_path',
    'models',
]


def parse_predict_config(config_obj, config_path):
    """parse [PREDICT] section of config.ini file

    Parameters
    ----------
    config_obj : ConfigParser
        containing config.ini file already loaded by parse function
    config_path : str
        path to config.ini file (used for error messages)

    Returns
    -------
    predict_config : vak.config.predict.PredictConfig
        instance of PredictConfig class that represents [PREDICT] section
        of config.ini file
    """
    predict_section = dict(
        config_obj['PREDICT'].items()
    )

    for required_option in REQUIRED_PREDICT_OPTIONS:
        if required_option not in predict_section:
            raise NoOptionError(
                f"the '{required_option}' option is required but was not found in the "
                f"PREDICT section of the config.ini file: {config_path}"
            )
    return PredictConfig(**predict_section)
