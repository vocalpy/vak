"""Function called by command-line interface for prep command"""

from __future__ import annotations

import pathlib
import shutil
import warnings

import tomlkit

from .. import config
from .. import prep as prep_module
from ..config.load import _load_toml_from_path
from ..config.validators import are_tables_valid


def purpose_from_toml(
    config_dict: dict, toml_path: str | pathlib.Path | None = None
) -> str:
    """Determine "purpose" from toml config,
    i.e., the command that will be run after we ``prep`` the data.

    By convention this is the other top-level table in the config file
    that correspond to a cli command besides ``[vak.prep]``, e.g. ``[vak.train]``.
    """
    # validate, make sure there aren't multiple commands in one config file first
    are_tables_valid(config_dict, toml_path=toml_path)
    config_dict = config_dict

    from ..cli.cli import CLI_COMMANDS  # avoid circular imports

    commands_that_are_not_prep = [
        command for command in CLI_COMMANDS if command != "prep"
    ]
    purpose = None
    for table_name in commands_that_are_not_prep:
        if table_name in config_dict:
            purpose = (
                table_name  # this top-level table is the "purpose" of the file
            )
    if purpose is None:
        raise ValueError(
            "Did not find a top-level table in configuration file that corresponds to a CLI command. "
            f"Configuration file path: {toml_path}\n"
            f"Found the following top-level tables: {config_dict.keys()}\n"
            f"Valid CLI commands besides ``prep`` (that correspond top-level tables) are: {commands_that_are_not_prep}"
        )
    return purpose


# note NO LOGGING -- we configure logger inside `core.prep`
# so we can save log file inside dataset directory

# see https://github.com/NickleDave/vak/issues/334
TABLES_PREP_SHOULD_PARSE = "prep"


def prep(toml_path):
    """Prepare datasets from vocalizations.
    Function called by command-line interface.

    Parameters
    ----------
    toml_path : str, pathlib.Path
        path to a configuration file in TOML format.
        Used to rewrite file with options determined by this function and needed for other functions

    Notes
    -----
    Saves a .csv file representing the dataset generated from data_dir.

    Datasets are used to train neural networks that segment audio files into
    vocalizations, and then predict labels for those segments.
    The function also prepares datasets so neural networks can predict the
    segmentation and annotation of vocalizations in them.
    It can also split a dataset into training, validation, and test sets,
    e.g. for benchmarking different neural network architectures.

    If the 'purpose' is set to 'train' or 'learncurve', and/or
    the duration of either the training or test set is provided,
    then the function attempts to split the dataset into training and test sets.
    A duration can also be specified for a validation set
    (used to measure performance during training).
    In these cases, the 'split' column in the .csv
    identifies which files (rows) belong to the training, test, and
    validation sets created from that Dataset.

    If the 'purpose' is set to 'predict' or 'eval',
    or no durations for any of the training sets are specified,
    then the function assumes all the vocalizations constitute a single
    dataset, and for all rows the 'split' columns for that dataset
    will be 'predict' or 'test' (respectively).
    """
    toml_path = pathlib.Path(toml_path)

    # open here because need to check whether the `dataset` already has a `path`, see #314 & #333
    config_dict = _load_toml_from_path(toml_path)

    # ---- figure out purpose of config file from tables; will save path of prep'd dataset in that table ---------------
    purpose = purpose_from_toml(config_dict, toml_path)
    if (
        "dataset" in config_dict[purpose]
        and "path" in config_dict[purpose]["dataset"]
    ):
        raise ValueError(
            f"This configuration file already has a '{purpose}.dataset' table with a 'path' key, "
            f"and running `prep` would overwrite the value for that key. To `prep` a new dataset, please "
            "either create a new configuration file, or remove "
            f"the 'path' key-value pair from the '{purpose}.dataset' table in the file:\n{toml_path}"
        )

    # now that we've checked that, go ahead and parse just the prep tabel;
    # we don't load the 'purpose' table into a config, to avoid error messages like non-existent paths, etc.
    # see https://github.com/NickleDave/vak/issues/334 and https://github.com/NickleDave/vak/issues/314
    cfg = config.Config.from_toml_path(
        toml_path, tables_to_parse=TABLES_PREP_SHOULD_PARSE
    )
    if cfg.prep is None:
        raise ValueError(
            f"prep called with a config.toml file that does not have a [vak.prep] table: {toml_path}"
        )

    if purpose == "predict":
        if cfg.prep.labelset is not None:
            warnings.warn(
                "config has a [vak.predict] table, but labelset option is specified in [vak.prep] table."
                "This would cause an error because the dataframe.from_files method will attempt to "
                f"check whether the files in the data_dir ({cfg.prep.data_dir}) have labels in "
                "labelset, even though those files don't have annotation.\n"
                "Setting labelset to None."
            )
            cfg.prep.labelset = None

    _, dataset_path = prep_module.prep(
        data_dir=cfg.prep.data_dir,
        purpose=purpose,
        dataset_type=cfg.prep.dataset_type,
        input_type=cfg.prep.input_type,
        audio_format=cfg.prep.audio_format,
        spect_format=cfg.prep.spect_format,
        spect_params=cfg.prep.spect_params,
        annot_format=cfg.prep.annot_format,
        annot_file=cfg.prep.annot_file,
        labelset=cfg.prep.labelset,
        audio_dask_bag_kwargs=cfg.prep.audio_dask_bag_kwargs,
        output_dir=cfg.prep.output_dir,
        train_dur=cfg.prep.train_dur,
        val_dur=cfg.prep.val_dur,
        test_dur=cfg.prep.test_dur,
        train_set_durs=cfg.prep.train_set_durs,
        num_replicates=cfg.prep.num_replicates,
    )

    # we re-open config using tomlkit so we can add path to dataset table in style-preserving way
    with toml_path.open("r") as fp:
        tomldoc = tomlkit.load(fp)
    if "dataset" not in tomldoc["vak"][purpose]:
        dataset_table = tomlkit.table()
        tomldoc["vak"][purpose].add("dataset", dataset_table)
    tomldoc["vak"][purpose]["dataset"].add("path", str(dataset_path))
    with toml_path.open("w") as fp:
        tomlkit.dump(tomldoc, fp)

    # lastly, copy config to dataset directory root
    shutil.copy(src=toml_path, dst=dataset_path)
