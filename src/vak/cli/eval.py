"""Evaluate a trained model with dataset specified in config.toml file."""

from __future__ import annotations

import logging
import pathlib

from .. import config
from .. import eval as eval_module
from ..common.logging import config_logging_for_cli, log_version

logger = logging.getLogger(__name__)


def eval(toml_path: str | pathlib.Path) -> None:
    """Evaluate a trained model with dataset specified in config.toml file.

    Function called by command-line interface.

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format.

    Returns
    -------
    None
    """
    toml_path = pathlib.Path(toml_path)
    cfg = config.Config.from_toml_path(toml_path)

    if cfg.eval is None:
        raise ValueError(
            f"eval called with a config.toml file that does not have a EVAL section: {toml_path}"
        )

    # ---- set up logging ---------------------------------------------------------------------------------------------
    config_logging_for_cli(
        log_dst=cfg.eval.output_dir, log_stem="eval", level="INFO", force=True
    )
    log_version(logger)

    logger.info("Logging results to {}".format(cfg.eval.output_dir))

    if cfg.eval.dataset.path is None:
        raise ValueError(
            "No value is specified for 'dataset_path' in this .toml config file."
            f"To generate a .csv file that represents the dataset, "
            f"please run the following command:\n'vak prep {toml_path}'"
        )

    eval_module.eval(
        model_config=cfg.eval.model.asdict(),
        dataset_config=cfg.eval.dataset.asdict(),
        trainer_config=cfg.eval.trainer.asdict(),
        checkpoint_path=cfg.eval.checkpoint_path,
        labelmap_path=cfg.eval.labelmap_path,
        output_dir=cfg.eval.output_dir,
        num_workers=cfg.eval.num_workers,
        batch_size=cfg.eval.batch_size,
        frames_standardizer_path=cfg.eval.frames_standardizer_path,
        post_tfm_kwargs=cfg.eval.post_tfm_kwargs,
    )
