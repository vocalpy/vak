from datetime import datetime
from pathlib import Path

from .. import config
from .. import core
from .. import logging


def predict(toml_path):
    """make predictions on dataset with trained model specified in config.toml file.
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
    cfg = config.parse.from_toml(toml_path)

    if cfg.predict is None:
        raise ValueError(
            f'predict called with a config.toml file that does not have a PREDICT section: {toml_path}'
        )

    # ---- set up logging ----------------------------------------------------------------------------------------------
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    logger = logging.get_logger(log_dst=cfg.prep.output_dir,
                                caller='predict',
                                timestamp=timenow,
                                logger_name=__name__)
    logger.info('Logging results to {}'.format(cfg.prep.output_dir))

    model_config_map = config.models.map_from_path(toml_path, cfg.predict.models)

    core.predict(csv_path=cfg.predict.csv_path,
                 checkpoint_path=cfg.predict.checkpoint_path,
                 labelmap_path=cfg.predict.labelmap_path,
                 model_config_map=model_config_map,
                 window_size=cfg.dataloader.window_size,
                 num_workers=cfg.predict.num_workers,
                 spect_key=cfg.spect_params.spect_key,
                 timebins_key=cfg.spect_params.timebins_key,
                 spect_scaler_path=cfg.predict.spect_scaler_path,
                 device=cfg.predict.device,
                 annot_csv_filename=cfg.predict.annot_csv_filename,
                 output_dir=cfg.predict.output_dir,
                 min_segment_dur=cfg.predict.min_segment_dur,
                 majority_vote=cfg.predict.majority_vote,
                 logger=logger)
