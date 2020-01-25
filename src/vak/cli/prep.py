import logging
import os
from pathlib import Path
from datetime import datetime

import toml

from .. import config
from ..io import dataframe
from ..util import train_test_dur_split


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
    If durations were specified for validation and test sets, then the .csv
    has a column representing which files belong to the training, test, and
    validation sets created from that Dataset.

    Datasets are used to train neural networks that segment audio files into
    vocalizations, and then predict labels for those segments.
    The function also prepares datasets so neural networks can predict the
    segmentation and annotation of vocalizations in them.
    It can also split a dataset into training, validation, and test sets,
    e.g. for benchmarking different neural network architectures.

    If no durations for any of the training sets are specified, then the
    function assumes all the vocalizations constitute a single training
    dataset. If the duration of either the training or test set is provided,
    then the function attempts to split the dataset into training and test sets.
    """
    toml_path = Path(toml_path)
    cfg = config.parse.from_toml(toml_path, sections=['PREP', 'SPECTROGRAM', 'DATALOADER'])

    if cfg.prep is None:
        raise ValueError(
            f'prep called with a config.ini file that does not have a PREP section: {toml_path}'
        )

    # pre-conditions ---------------------------------------------------------------------------------------------------
    if cfg.prep.audio_format is None and cfg.prep.spect_format is None:
        raise ValueError("Must specify either audio_format or spect_format")

    if cfg.prep.audio_format and cfg.prep.spect_format:
        raise ValueError("Cannot specify both audio_format and spect_format, "
                         "unclear whether to create spectrograms from audio files or "
                         "use already-generated spectrograms from array files")

    if cfg.prep.labelset is not None:
        if type(cfg.prep.labelset) not in (set, list):
            raise TypeError(
                f"type of labelset must be set or list, but type was: {type(labelset)}"
            )

        if type(cfg.prep.labelset) == list:
            labelset_set = set(cfg.prep.labelset)
            if len(cfg.prep.labelset) != len(labelset_set):
                raise ValueError(
                    'labelset contains repeated elements, should be a set (i.e. all members unique.\n'
                    f'Labelset was: {cfg.prep.labelset}'
                )
            else:
                cfg.prep.labelset = labelset_set

    if cfg.prep.output_dir:
        if not os.path.isdir(cfg.prep.output_dir):
            raise NotADirectoryError(
                f'output_dir not found: {cfg.prep.output_dir}'
            )
    elif cfg.prep.output_dir is None:
        cfg.prep.output_dir = cfg.prep.data_dir

    # ---- logging -----------------------------------------------------------------------------------------------------
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    # ---- figure out what section we will be saving path in, and prefix of option name in that section ----------------
    # (e.g., if it's PREDICT section then the prefix will be 'predict' for 'predict_vds_path' option
    with toml_path.open('r') as fp:
        config_toml = toml.load(fp)
    if 'TRAIN' in config_toml:
        section = 'TRAIN'
    elif 'LEARNCURVE' in config_toml:
        section = 'LEARNCURVE'
    elif 'PREDICT' in config_toml:
        section = 'PREDICT'
    else:
        raise ValueError(
            'Did not find a section named TRAIN, LEARNCURVE, or PREDICT in config.ini file;'
            ' unable to determine which section to add paths to prepared datasets to'
        )

    # ---- figure out file name ----------------------------------------------------------------------------------------
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    _, tail = os.path.split(cfg.prep.data_dir)
    csv_fname_stem = f'{tail}_prep_{timenow}'
    csv_path = os.path.join(cfg.prep.output_dir, f'{csv_fname_stem}.csv')

    # ---- figure out if we're going to split into train / val / test sets ---------------------------------------------
    if all([dur is None for dur in (cfg.prep.train_dur,
                                    cfg.prep.val_dur,
                                    cfg.prep.test_dur)]):
        # then we're not going to split
        do_split = False
    else:
        if cfg.prep.val_dur is not None and cfg.prep.train_dur is None and cfg.prep.test_dur is None:
            raise ValueError('cannot specify only val_dur, unclear how to split dataset into training and test sets')
        else:
            do_split = True

    # ---- actually make the dataset -----------------------------------------------------------------------------------
    vak_df = dataframe.from_files(labelset=cfg.prep.labelset,
                                  data_dir=cfg.prep.data_dir,
                                  annot_format=cfg.prep.annot_format,
                                  output_dir=cfg.prep.output_dir,
                                  annot_file=cfg.prep.annot_file,
                                  audio_format=cfg.prep.audio_format,
                                  spect_format=cfg.prep.spect_format,
                                  spect_params=cfg.spect_params)

    if do_split:
        # save before splitting, jic duration args are not valid (we can't know until we make dataset)
        vak_df.to_csv(csv_path)
        vak_df = train_test_dur_split(vak_df,
                                      labelset=cfg.prep.labelset,
                                      train_dur=cfg.prep.train_dur,
                                      val_dur=cfg.prep.val_dur,
                                      test_dur=cfg.prep.test_dur)

    elif do_split is False:
        # add a split column, but assign everything to the same 'split'
        vak_df = dataframe.add_split_col(vak_df, split=section.lower())

    vak_df.to_csv(csv_path)

    # use config and section from above to add csv_path to config.ini file
    config_toml[section]['csv_path'] = csv_path

    with toml_path.open('w') as fp:
        toml.dump(config_toml, fp)
