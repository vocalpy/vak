from pathlib import Path
from datetime import datetime
import warnings

import toml

from .. import config
from .. import core
from .. import logging


def prep(toml_path):
    """prepare datasets from vocalizations.
    Function called by command-line interface.

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format.
        Used to rewrite file with options determined by this function and needed for other functions

    Returns
    -------
    None

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
    cfg = config.parse.from_toml(toml_path)

    if cfg.prep is None:
        raise ValueError(
            f'prep called with a config.toml file that does not have a PREP section: {toml_path}'
        )

    # ---- set up logging ----------------------------------------------------------------------------------------------
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    logger = logging.get_logger(log_dst=cfg.prep.output_dir,
                                caller='prep',
                                timestamp=timenow,
                                logger_name=__name__)

    # ---- figure out purpose of config file from sections; will save csv path in that section -------------------------
    with toml_path.open('r') as fp:
        config_toml = toml.load(fp)

    if 'EVAL' in config_toml:
        section = 'EVAL'
    elif 'LEARNCURVE' in config_toml:
        section = 'LEARNCURVE'
    elif 'PREDICT' in config_toml:
        section = 'PREDICT'
        if cfg.prep.labelset is not None:
            warnings.warn(
                "config has a PREDICT section, but labelset option is specified in PREP section."
                "This would cause an error because the dataframe.from_files section will attempt to "
                f"check whether the files in the data_dir ({cfg.prep.data_dir}) have labels in "
                "labelset, even though those files don't have annotation.\n"
                "Setting labelset to None."
            )
            cfg.prep.labelset = None
    elif 'TRAIN' in config_toml:
        section = 'TRAIN'
    else:
        raise ValueError(
            'Did not find a section named TRAIN, LEARNCURVE, or PREDICT in config.toml file;'
            ' unable to determine which section to add paths to prepared datasets to'
        )
    logger.info(f'determined that config file has section: {section}\nWill add csv_path option to that section')

    purpose = section.lower()
    vak_df, csv_path = core.prep(data_dir=cfg.prep.data_dir,
                                 purpose=purpose,
                                 audio_format=cfg.prep.audio_format,
                                 spect_format=cfg.prep.spect_format,
                                 spect_output_dir=cfg.prep.spect_output_dir,
                                 spect_params=cfg.spect_params,
                                 annot_format=cfg.prep.annot_format,
                                 annot_file=cfg.prep.annot_file,
                                 labelset=cfg.prep.labelset,
                                 output_dir=cfg.prep.output_dir,
                                 train_dur=cfg.prep.train_dur,
                                 val_dur=cfg.prep.val_dur,
                                 test_dur=cfg.prep.test_dur,
                                 logger=logger,
                                 )

    # use config and section from above to add csv_path to config.toml file
    config_toml[section]['csv_path'] = str(csv_path)

    with toml_path.open('w') as fp:
        toml.dump(config_toml, fp)
