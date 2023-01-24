from pathlib import Path
import logging
import warnings

import toml

from .. import (
    config,
    core
)
from ..config.parse import _load_toml_from_path
from ..config.validators import are_sections_valid
from ..logging import config_logging_for_cli, log_version


logger = logging.getLogger(__name__)


def purpose_from_toml(config_toml, toml_path=None):
    """determine "purpose" from toml config,
    i.e., the command that will be run after we ``prep`` the data.

    By convention this is the other section in the config file
    that correspond to a cli command besides '[PREP]'
    """
    # validate, make sure there aren't multiple commands in one config file first
    are_sections_valid(config_toml, toml_path=toml_path)

    from ..cli.cli import CLI_COMMANDS  # avoid circular imports

    commands_that_are_not_prep = (
        command for command in CLI_COMMANDS if command != "prep"
    )
    for command in commands_that_are_not_prep:
        section_name = (
            command.upper()
        )  # we write section names in uppercase, e.g. `[PREP]`, by convention
        if section_name in config_toml:
            return section_name.lower()  # this is the "purpose" of the file


# see https://github.com/NickleDave/vak/issues/334
SECTIONS_PREP_SHOULD_PARSE = ("PREP", "SPECT_PARAMS")


def prep(toml_path):
    """Prepare datasets from vocalizations.
    Function called by command-line interface.

    Parameters
    ----------
    toml_path : str, Path
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
    toml_path = Path(toml_path)

    # open here because need to check for `csv_path` in this function, see #314 & #333
    config_toml = _load_toml_from_path(toml_path)
    # ---- figure out purpose of config file from sections; will save csv path in that section -------------------------
    purpose = purpose_from_toml(config_toml, toml_path)
    if (
        "csv_path" in config_toml[purpose.upper()]
        and config_toml[purpose.upper()]["csv_path"] is not None
    ):
        raise ValueError(
            f"config .toml file already has a 'csv_path' option in the '{purpose.upper()}' section, "
            f"and running `prep` would overwrite that value. To `prep` a new dataset, please remove "
            f"the 'csv_path' option from the '{purpose.upper()}' section in the config file:\n{toml_path}"
        )

    # now that we've checked that, go ahead and parse the sections we want
    cfg = config.parse.from_toml_path(toml_path, sections=SECTIONS_PREP_SHOULD_PARSE)
    # notice we ignore any other option/values in the 'purpose' section,
    # see https://github.com/NickleDave/vak/issues/334 and https://github.com/NickleDave/vak/issues/314
    if cfg.prep is None:
        raise ValueError(
            f"prep called with a config.toml file that does not have a PREP section: {toml_path}"
        )

    if purpose == "predict":
        if cfg.prep.labelset is not None:
            warnings.warn(
                "config has a PREDICT section, but labelset option is specified in PREP section."
                "This would cause an error because the dataframe.from_files section will attempt to "
                f"check whether the files in the data_dir ({cfg.prep.data_dir}) have labels in "
                "labelset, even though those files don't have annotation.\n"
                "Setting labelset to None."
            )
            cfg.prep.labelset = None

    # ---- set up logging ----------------------------------------------------------------------------------------------
    config_logging_for_cli(
        log_dst=cfg.prep.output_dir,
        log_stem="prep",
        level="INFO",
        force=True
    )
    log_version(logger)

    section = purpose.upper()
    logger.info(
        f"Determined that purpose of config file is: {purpose}.\n"
        f"Will add 'csv_path' option to '{section}' section."
    )

    vak_df, csv_path = core.prep(
        data_dir=cfg.prep.data_dir,
        purpose=purpose,
        audio_format=cfg.prep.audio_format,
        spect_format=cfg.prep.spect_format,
        spect_output_dir=cfg.prep.spect_output_dir,
        spect_params=cfg.spect_params,
        annot_format=cfg.prep.annot_format,
        annot_file=cfg.prep.annot_file,
        labelset=cfg.prep.labelset,
        audio_dask_bag_kwargs=cfg.prep.audio_dask_bag_kwargs,
        output_dir=cfg.prep.output_dir,
        train_dur=cfg.prep.train_dur,
        val_dur=cfg.prep.val_dur,
        test_dur=cfg.prep.test_dur,
    )

    # use config and section from above to add csv_path to config.toml file
    config_toml[section]["csv_path"] = str(csv_path)

    with toml_path.open("w") as fp:
        toml.dump(config_toml, fp)
