from pathlib import Path
import shutil

from .. import config
from .. import core
from .. import logging
from ..paths import generate_results_dir_name_as_path
from ..timenow import get_timenow_as_str


def train_checkpoint(toml_path):
    """train models using training set specified in config.toml file.
    Starts from a checkpoint given in config.toml file.
    Function called by command-line interface.
    Updated by K.L.Provost 8 Dec 2021

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format.

    Returns
    -------
    None

    Trains models from checkpoints, saves results in new directory within root_results_dir specified
    in config.toml file, and adds path to that new directory to config.toml file.
    """
    toml_path = Path(toml_path)
    cfg = config.parse.from_toml_path(toml_path)

    if cfg.train_checkpoint is None:
        raise ValueError(
            f"train_checkpoint called with a config.toml file that does not have a TRAIN_CHECKPOINT section: {toml_path}"
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    results_path = generate_results_dir_name_as_path(cfg.train_checkpoint.root_results_dir)
    results_path.mkdir(parents=True)
    # copy config file into results dir now that we've made the dir
    shutil.copy(toml_path, results_path)

    # ---- set up logging ----------------------------------------------------------------------------------------------
    logger = logging.get_logger(
        log_dst=results_path,
        caller="train_checkpoint",
        timestamp=get_timenow_as_str(),
        logger_name=__name__,
    )
    logger.info("Logging results to {}".format(results_path))

    model_config_map = config.models.map_from_path(toml_path, cfg.train_checkpoint.models)

    core.train_checkpoint(
        model_config_map=model_config_map,
        csv_path=cfg.train_checkpoint.csv_path,
        labelset=cfg.prep.labelset,
        window_size=cfg.dataloader.window_size,
        batch_size=cfg.train_checkpoint.batch_size,
        num_epochs=cfg.train_checkpoint.num_epochs,
        num_workers=cfg.train_checkpoint.num_workers,
        checkpoint_path=cfg.train_checkpoint.checkpoint_path,
        labelmap_path=cfg.train_checkpoint.labelmap_path,
        spect_scaler_path=cfg.train_checkpoint.spect_scaler_path,
        results_path=results_path,
        spect_key=cfg.spect_params.spect_key,
        timebins_key=cfg.spect_params.timebins_key,
        normalize_spectrograms=cfg.train_checkpoint.normalize_spectrograms,
        shuffle=cfg.train_checkpoint.shuffle,
        val_step=cfg.train_checkpoint.val_step,
        ckpt_step=cfg.train_checkpoint.ckpt_step,
        patience=cfg.train_checkpoint.patience,
        device=cfg.train_checkpoint.device,
        logger=logger,
    )
