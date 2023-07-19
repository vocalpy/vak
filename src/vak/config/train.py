"""parses [TRAIN] section of config"""
import attr
from attr import converters, validators
from attr.validators import instance_of

from .validators import is_valid_model_name
from ..common import device
from ..common.converters import bool_from_str, expanded_user_path


@attr.s
class TrainConfig:
    """class that represents [TRAIN] section of config.toml file

    Attributes
    ----------
    model : str
        Model name, e.g., ``model = "TweetyNet"``
    dataset_path : str
        Path to dataset, e.g., a csv file generated by running ``vak prep``.
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
    checkpoint_path : str
        path to directory with checkpoint files saved by Torch, to reload model. 
        Default is None, in which case a new model is initialized. 
    spect_scaler_path : str
        path to a saved SpectScaler object used to normalize spectrograms.
        If spectrograms were normalized and this is not provided, will give
        incorrect results. Default is None.
    """
    # required
    model = attr.ib(
        validator=[instance_of(str), is_valid_model_name],
    )
    num_epochs = attr.ib(converter=int, validator=instance_of(int))
    batch_size = attr.ib(converter=int, validator=instance_of(int))
    root_results_dir = attr.ib(converter=expanded_user_path)

    # optional
    # dataset_path is actually 'required' but we can't enforce that here because cli.prep looks at
    # what sections are defined to figure out where to add dataset_path after it creates the csv
    dataset_path = attr.ib(
        converter=converters.optional(expanded_user_path),
        default=None,
    )

    results_dirname = attr.ib(
        converter=converters.optional(expanded_user_path),
        default=None,
    )

    normalize_spectrograms = attr.ib(
        converter=bool_from_str,
        validator=validators.optional(instance_of(bool)),
        default=False,
    )

    num_workers = attr.ib(validator=instance_of(int), default=2)
    device = attr.ib(validator=instance_of(str), default=device.get_default())
    shuffle = attr.ib(
        converter=bool_from_str, validator=instance_of(bool), default=True
    )

    val_step = attr.ib(
        converter=converters.optional(int),
        validator=validators.optional(instance_of(int)),
        default=None,
    )
    ckpt_step = attr.ib(
        converter=converters.optional(int),
        validator=validators.optional(instance_of(int)),
        default=None,
    )
    patience = attr.ib(
        converter=converters.optional(int),
        validator=validators.optional(instance_of(int)),
        default=None,
    )
    checkpoint_path = attr.ib(
        converter=converters.optional(expanded_user_path),
        default=None,
    )
    spect_scaler_path = attr.ib(
        converter=converters.optional(expanded_user_path),
        default=None,
    )

    train_dataset_params = attr.ib(
        converter=converters.optional(dict),
        validator=validators.optional(instance_of(dict)),
        default=None,
    )
    val_dataset_params = attr.ib(
        converter=converters.optional(dict),
        validator=validators.optional(instance_of(dict)),
        default=None,
    )
