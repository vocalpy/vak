from pathlib import Path
import shutil
from datetime import datetime

from .. import config
from .. import core
from .. import logging


def learning_curve(toml_path):
    """generate learning curve, by training models on training sets across a
    range of sizes and then measure accuracy of those models on a test set.
    Function called by command-line interface.

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format.

    Returns
    -------
    None

    Trains models, saves results in new directory within root_results_dir specified
    in config.toml file, and adds path to that new directory to config.toml file.
    """
    toml_path = Path(toml_path)
    cfg = config.parse.from_toml(toml_path)

    if cfg.learncurve is None:
        raise ValueError(
            f'train called with a config.toml file that does not have a TRAIN section: {toml_path}'
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    results_dirname = f'results_{timenow}'
    if cfg.learncurve.root_results_dir:
        results_path = Path(cfg.learncurve.root_results_dir)
    else:
        results_path = Path('.')
    results_path = results_path.joinpath(results_dirname)
    results_path.mkdir(parents=True)
    # copy config file into results dir now that we've made the dir
    shutil.copy(toml_path, results_path)

    # ---- set up logging ----------------------------------------------------------------------------------------------
    logger = logging.get_logger(log_dst=results_path,
                                caller='train',
                                timestamp=timenow,
                                logger_name=__name__)
    logger.info('Logging results to {}'.format(results_path))

    model_config_map = config.models.map_from_path(toml_path, cfg.learncurve.models)

    core.learning_curve(model_config_map,
                        train_set_durs=cfg.learncurve.train_set_durs,
                        num_replicates=cfg.learncurve.num_replicates,
                        csv_path=cfg.learncurve.csv_path,
                        labelset=cfg.prep.labelset,
                        window_size=cfg.dataloader.window_size,
                        batch_size=cfg.learncurve.batch_size,
                        num_epochs=cfg.learncurve.num_epochs,
                        num_workers=cfg.learncurve.num_workers,
                        results_path=results_path,
                        previous_run_path=cfg.learncurve.previous_run_path,
                        spect_key=cfg.spect_params.spect_key,
                        timebins_key=cfg.spect_params.timebins_key,
                        normalize_spectrograms=cfg.learncurve.normalize_spectrograms,
                        shuffle=cfg.learncurve.shuffle,
                        val_step=cfg.learncurve.val_step,
                        ckpt_step=cfg.learncurve.ckpt_step,
                        patience=cfg.learncurve.patience,
                        device=cfg.learncurve.device,
                        logger=logger,
                        )
