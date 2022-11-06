import logging
from pathlib import Path
import shutil

from .. import (
    config,
    core
)
from ..logging import config_logging_for_cli, log_version
from ..paths import generate_results_dir_name_as_path


logger = logging.getLogger(__name__)


def train(toml_path):
    """train models using training set specified in config.toml file.
    Function called by command-line interface.

    Trains models, saves results in new directory within root_results_dir specified
    in config.toml file, and adds path to that new directory to config.toml file.

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format.
    """
    toml_path = Path(toml_path)
    cfg = config.parse.from_toml_path(toml_path)

    if cfg.train is None:
        raise ValueError(
            f"train called with a config.toml file that does not have a TRAIN section: {toml_path}"
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    results_path = generate_results_dir_name_as_path(cfg.train.root_results_dir)
    results_path.mkdir(parents=True)
    # copy config file into results dir now that we've made the dir
    shutil.copy(toml_path, results_path)

    # ---- set up logging ----------------------------------------------------------------------------------------------
    config_logging_for_cli(
        log_dst=results_path,
        log_stem="train",
        level="INFO",
        force=True
    )
    log_version(logger)
    logger.info("Logging results to {}".format(results_path))

    model_config_map = config.models.map_from_path(toml_path, cfg.train.models)

    if cfg.train.csv_path is None:
        raise ValueError(
            "No value is specified for 'csv_path' in this .toml config file."
            f"To generate a .csv file that represents the dataset, "
            f"please run the following command:\n'vak prep {toml_path}'"
        )

    if cfg.train.labelmap_path is not None:
        labelset, labelmap_path = None, cfg.train.labelmap_path
    else:
        labelset, labelmap_path = cfg.prep.labelset, None

    core.train(
        model_config_map=model_config_map,
        csv_path=cfg.train.csv_path,
        labelset=labelset,
        window_size=cfg.dataloader.window_size,
        batch_size=cfg.train.batch_size,
        num_epochs=cfg.train.num_epochs,
        num_workers=cfg.train.num_workers,
        checkpoint_path=cfg.train.checkpoint_path,
        spect_scaler_path=cfg.train.spect_scaler_path,
        labelmap_path=labelmap_path,
        results_path=results_path,
        spect_key=cfg.spect_params.spect_key,
        timebins_key=cfg.spect_params.timebins_key,
        normalize_spectrograms=cfg.train.normalize_spectrograms,
        shuffle=cfg.train.shuffle,
        val_step=cfg.train.val_step,
        ckpt_step=cfg.train.ckpt_step,
        patience=cfg.train.patience,
        device=cfg.train.device,
    )
