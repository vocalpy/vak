import logging
from pathlib import Path

from .. import (
    config,
    core,
)
from ..logging import config_logging_for_cli, log_version


logger = logging.getLogger(__name__)


def eval(toml_path):
    """evaluate a trained model with dataset specified in config.toml file.
    Function called by command-line interface.

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format.

    Returns
    -------
    None
    """
    toml_path = Path(toml_path)
    cfg = config.parse.from_toml_path(toml_path)

    if cfg.eval is None:
        raise ValueError(
            f"eval called with a config.toml file that does not have a EVAL section: {toml_path}"
        )

    # ---- set up logging ---------------------------------------------------------------------------------------------
    config_logging_for_cli(
        log_dst=cfg.eval.output_dir,
        log_stem="eval",
        level="INFO",
        force=True
    )
    log_version(logger)

    logger.info("Logging results to {}".format(cfg.eval.output_dir))

    model_config_map = config.models.map_from_path(toml_path, cfg.eval.models)

    if cfg.eval.csv_path is None:
        raise ValueError(
            "No value is specified for 'csv_path' in this .toml config file."
            f"To generate a .csv file that represents the dataset, "
            f"please run the following command:\n'vak prep {toml_path}'"
        )

    core.eval(
        cfg.eval.csv_path,
        model_config_map,
        checkpoint_path=cfg.eval.checkpoint_path,
        labelmap_path=cfg.eval.labelmap_path,
        output_dir=cfg.eval.output_dir,
        window_size=cfg.dataloader.window_size,
        num_workers=cfg.eval.num_workers,
        spect_scaler_path=cfg.eval.spect_scaler_path,
        spect_key=cfg.spect_params.spect_key,
        timebins_key=cfg.spect_params.timebins_key,
        device=cfg.eval.device,
        post_tfm_kwargs=cfg.eval.post_tfm_kwargs,
    )
