import logging
import shutil
from pathlib import Path

from .. import config, learncurve
from ..common.logging import config_logging_for_cli, log_version
from ..common.paths import generate_results_dir_name_as_path

logger = logging.getLogger(__name__)


def learning_curve(toml_path):
    """generate learning curve, by training models on training sets across a
    range of sizes and then measure accuracy of those models on a test set.
    Function called by command-line interface.

    Trains models, saves results in new directory within root_results_dir specified
    in config.toml file, and adds path to that new directory to config.toml file.

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format.
    """
    toml_path = Path(toml_path)
    cfg = config.Config.from_toml_path(toml_path)

    if cfg.learncurve is None:
        raise ValueError(
            f"learncurve called with a config.toml file that does not have a LEARNCURVE section: {toml_path}"
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    results_path = generate_results_dir_name_as_path(
        cfg.learncurve.root_results_dir
    )
    results_path.mkdir(parents=True)
    # copy config file into results dir now that we've made the dir
    shutil.copy(toml_path, results_path)

    # ---- set up logging ----------------------------------------------------------------------------------------------
    config_logging_for_cli(
        log_dst=results_path, log_stem="learncurve", level="INFO", force=True
    )
    log_version(logger)
    logger.info("Logging results to {}".format(results_path))

    if cfg.learncurve.dataset.path is None:
        raise ValueError(
            "No value is specified for 'dataset_path' in this .toml config file."
            f"To generate a .csv file that represents the dataset, "
            f"please run the following command:\n'vak prep {toml_path}'"
        )

    learncurve.learning_curve(
        model_config=cfg.learncurve.model.asdict(),
        dataset_config=cfg.learncurve.dataset.asdict(),
        trainer_config=cfg.learncurve.trainer.asdict(),
        batch_size=cfg.learncurve.batch_size,
        num_epochs=cfg.learncurve.num_epochs,
        num_workers=cfg.learncurve.num_workers,
        results_path=results_path,
        post_tfm_kwargs=cfg.learncurve.post_tfm_kwargs,
        standardize_frames=cfg.learncurve.standardize_frames,
        shuffle=cfg.learncurve.shuffle,
        val_step=cfg.learncurve.val_step,
        ckpt_step=cfg.learncurve.ckpt_step,
        patience=cfg.learncurve.patience,
    )
