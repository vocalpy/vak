from functools import partial
import json
import logging
from pathlib import Path
import shutil
import sys
from datetime import datetime

import joblib
import pandas as pd
import torch.utils.data
from torchvision import transforms

from .. import config
from .. import models
from .. import util
from ..datasets.spectrogram_window_dataset import SpectrogramWindowDataset
from ..io import dataframe
from ..transforms import StandardizeSpect


def train(toml_path):
    """train models using training set specified in config.toml file.
    Function called by command-line interface.

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format.

    Returns
    -------
    None

    Trains models, saves results in new directory within root_results_dir specified
    in config.ini file, and adds path to that new directory to config.ini file.
    """
    toml_path = Path(toml_path)
    cfg = config.parse.from_toml(toml_path)

    if cfg.train is None:
        raise ValueError(
            f'train called with a config.toml file that does not have a TRAIN section: {toml_path}'
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
    shutil.copy(toml_path, results_path)

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
        standardize = StandardizeSpect.fit_df(dataset_df, spect_key=cfg.spect_params.spect_key)
        joblib.dump(standardize,
                    results_path.joinpath('StandardizeSpect'))
    else:
        standardize = None

    # below, if we're going to train network to predict unlabeled segments, then
    # we need to include a class for those unlabeled segments in labelmap,
    # the mapping from labelset provided by user to a set of consecutive
    # integers that the network learns to predict
    has_unlabeled = util.dataset.has_unlabeled(cfg.train.csv_path, cfg.prep.labelset, cfg.spect_params.timebins_key)
    if has_unlabeled:
        map_unlabeled = True
    else:
        map_unlabeled = False
    labelmap = util.labels.to_map(cfg.prep.labelset, map_unlabeled=map_unlabeled)
    logger.debug(
        f'number of classes in labelmap: {len(labelmap)}'
    )
    # save labelmap in case we need it later
    with open(results_path.joinpath('labelmap.json'), 'w') as f:
        json.dump(labelmap, f)

    logger.info(f'using training dataset from {cfg.train.csv_path}')

    # make an "add channel" transform to use with Lambda
    # this way a spectrogram 'image' has a "channel" dimension (of size 1)
    # that convolutional layers can work on
    def to_floattensor(ndarray):
        return torch.from_numpy(ndarray).float()

    add_channel = partial(torch.unsqueeze, dim=0)
    transform = transforms.Compose(
        [transforms.Lambda(standardize),
         transforms.Lambda(to_floattensor),
         transforms.Lambda(add_channel)]
    )

    def to_longtensor(ndarray):
        return torch.from_numpy(ndarray).long()
    target_transform = transforms.Compose(
        [transforms.Lambda(to_longtensor)]
    )

    train_dataset = SpectrogramWindowDataset.from_csv(csv_path=cfg.train.csv_path,
                                                      split='train',
                                                      labelmap=labelmap,
                                                      window_size=cfg.dataloader.window_size,
                                                      spect_key=cfg.spect_params.spect_key,
                                                      timebins_key=cfg.spect_params.timebins_key,
                                                      transform=transform,
                                                      target_transform=target_transform
                                                      )
    train_data = torch.utils.data.DataLoader(dataset=train_dataset,
                                             shuffle=cfg.train.shuffle,
                                             batch_size=cfg.train.batch_size,
                                             num_workers=cfg.train.num_workers)
    train_dur = dataframe.split_dur(dataset_df, 'train')
    logger.info(
        f'Total duration of training set (in s): {train_dur}'
    )

    # ---------------- load validation set (if there is one) -----------------------------------------------------------
    if cfg.train.val_error_step:
        val_dataset = SpectrogramWindowDataset.from_csv(csv_path=cfg.train.csv_path,
                                                        split='val',
                                                        labelmap=labelmap,
                                                        window_size=cfg.dataloader.window_size,
                                                        spect_key=cfg.spect_params.spect_key,
                                                        timebins_key=cfg.spect_params.timebins_key,
                                                        transform=transform,
                                                        target_transform=target_transform
                                                        )
        val_data = torch.utils.data.DataLoader(dataset=val_dataset,
                                               shuffle=False,
                                               batch_size=cfg.train.batch_size,
                                               num_workers=cfg.train.num_workers)
        val_dur = dataframe.split_dur(dataset_df, 'val')
        logger.info(
            f'Total duration of validation set (in s): {val_dur}'
        )

        logger.info(
            f'will measure error on validation set every {cfg.train.val_error_step} steps of training'
        )
    else:
        val_data = None

    model_config_map = config.models.map_from_path(toml_path, cfg.train.models)
    models_map = models.from_model_config_map(
        model_config_map,
        num_classes=len(labelmap),
        input_shape=train_dataset.shape
    )
    for model_name, model in models_map.items():
        results_model_root = results_path.joinpath(model_name)
        results_model_root.mkdir()
        ckpt_root = results_model_root.joinpath('checkpoints')
        ckpt_root.mkdir()
        logger.info(
            f'training {model_name}'
        )
        model.fit(train_data=train_data,
                  num_epochs=cfg.train.num_epochs,
                  ckpt_root=ckpt_root,
                  val_data=val_data,
                  val_step=cfg.train.val_error_step,
                  checkpoint_step=cfg.train.checkpoint_step,
                  patience=cfg.train.patience,
                  single_ckpt=cfg.train.save_only_single_checkpoint_file,
                  device=cfg.train.device)
