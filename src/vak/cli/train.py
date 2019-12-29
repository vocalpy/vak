from configparser import ConfigParser
import json
import logging
from pathlib import Path
import shutil
import sys
from datetime import datetime

import joblib
import pandas as pd

from .. import config
from .. import core
from .. import models
from .. import utils
from ..io import dataset, dataframe
from ..utils.spect import SpectScaler


def train(config_path):
    """train models using training set specified in config.ini file.
    Function called by command-line interface.

    Parameters
    ----------
    config_path : str, Path
        path to config.ini file. Used to rewrite file with options determined by
        this function and needed for other functions

    Returns
    -------
    None

    Trains models, saves results in new directory within root_results_dir specified
    in config.ini file, and adds path to that new directory to config.ini file.
    """
    cfg = config.parse.from_path(config_path)

    if cfg.train is None:
        raise ValueError(
            f'train called with a config.ini file that does not have a TRAIN section: {config_path}'
        )

    dataset_df = pd.read_csv(cfg.train.csv_path)
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    if cfg.train.val_error_step and not dataset_df['split'].str.contains('val').any():
        raise ValueError(
            f"val_error_step set to {cfg.train.val_error_step} but dataset does not contain a validation set; "
            f"please run `vak prep` with a config.ini file that specifies a duration for the validation set."
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    results_dirname = f'results_{timenow}'
    if cfg.train.root_results_dir:
        results_path = Path(cfg.train.root_results_dir)
    else:
        results_path = Path('.')
    results_path = results_path.joinpath(results_dirname)
    results_path.mkdir(parents=True)
    # copy config file into results dir now that we've made the dir
    shutil.copy(config_path, results_path)

    # ---- set up logging ----------------------------------------------------------------------------------------------
    logfile_name = results_path.joinpath('logfile_from_train_' + timenow + '.log')
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
    logger.addHandler(logging.FileHandler(logfile_name))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info('Logging results to {}'.format(results_path))

    timebin_dur = dataframe.validate_and_get_timebin_dur(dataset_df)
    logger.info(
        f'Size of each timebin in spectrogram, in seconds: {timebin_dur}'
    )

    # ---------------- load training data  -----------------------------------------------------------------------------
    if cfg.train.normalize_spectrograms:
        logger.info('will normalize spectrograms')
        spect_scaler = SpectScaler.fit_df(dataset_df, spect_key=cfg.spect_params.spect_key)
        joblib.dump(spect_scaler,
                    results_path.joinpath('spect_scaler'))
    else:
        spect_scaler = None

    # below, if we're going to train network to predict unlabeled segments, then
    # we need to include a class for those unlabeled segments in labelmap,
    # the mapping from labelset provided by user to a set of consecutive
    # integers that the network learns to predict
    has_unlabeled = utils.dataset.has_unlabeled(cfg.train.csv_path, cfg.prep.labelset, cfg.spect_params.timebins_key)
    if has_unlabeled:
        map_unlabeled = True
    else:
        map_unlabeled = False
    labelmap = utils.labels.to_map(cfg.prep.labelset, map_unlabeled=map_unlabeled)
    logger.debug(
        f'number of classes in labelmap: {len(labelmap)}'
    )
    # save labelmap in case we need it later
    with open(results_path.joinpath('labelmap.json'), 'w') as f:
        json.dump(labelmap, f)

    logger.info(f'using training dataset from {cfg.train.csv_path}')
    train_loader = dataset.dataloaders.WindowDataLoader.from_csv(csv_path=cfg.train.csv_path,
                                                                 split='train',
                                                                 labelmap=labelmap,
                                                                 window_size=cfg.dataloader.window_size,
                                                                 batch_size=cfg.train.batch_size,
                                                                 shuffle=cfg.train.shuffle,
                                                                 spect_key=cfg.spect_params.spect_key,
                                                                 timebins_key=cfg.spect_params.timebins_key,
                                                                 spect_scaler=spect_scaler)

    train_dur = dataframe.split_dur(dataset_df, 'train')
    logger.info(
        f'Total duration of training set (in s): {train_dur}'
    )

    # ---------------- load validation set (if there is one) -----------------------------------------------------------
    if cfg.train.val_error_step:
        val_loader = dataset.dataloaders.WindowDataLoader.from_csv(csv_path=cfg.train.csv_path,
                                                                   split='val',
                                                                   labelmap=labelmap,
                                                                   window_size=cfg.dataloader.window_size,
                                                                   batch_size=cfg.train.batch_size,
                                                                   shuffle=False,
                                                                   spect_key=cfg.spect_params.spect_key,
                                                                   timebins_key=cfg.spect_params.timebins_key,
                                                                   spect_scaler=spect_scaler)

        val_dur = dataframe.split_dur(dataset_df, 'val')
        logger.info(
            f'Total duration of validation set (in s): {val_dur}'
        )

        logger.info(
            f'will measure error on validation set every {cfg.train.val_error_step} steps of training'
        )
    else:
        val_loader = None

    model_config_map = config.models.map_from_path(config_path, cfg.train.models)
    models_map = models.from_model_config_map(
        model_config_map,
        num_classes=len(labelmap),
        input_shape=train_loader.shape
    )

    core.train(models=models_map,
               optimizer=cfg.train.optimizer,
               loss=cfg.train.loss,
               metrics=cfg.train.metrics,
               num_epochs=cfg.train.num_epochs,
               train_loader=train_loader,
               val_loader=val_loader,
               val_error_step=cfg.train.val_error_step,
               checkpoint_step=cfg.train.checkpoint_step,
               patience=cfg.train.patience,
               save_only_single_checkpoint_file=cfg.train.save_only_single_checkpoint_file,
               results_path=results_path,
               )

    # lastly rewrite config file,
    # so that paths where results were saved are automatically in config
    config_obj = ConfigParser()
    config_obj.read(config_path)
    config_obj.set(section='TRAIN',
                   option='results_dir_made_by_main_script',
                   value=results_dirname)
    with open(config_path, 'w') as fp:
        config_obj.write(fp)
