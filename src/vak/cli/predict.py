import logging
from pathlib import Path

from .. import common, config
from .. import predict as predict_module
from ..common.logging import config_logging_for_cli, log_version

logger = logging.getLogger(__name__)


def predict(toml_path):
    """make predictions on dataset with trained model specified in config.toml file.
    Function called by command-line interface.

    Parameters
    ----------
    toml_path : str, Path
        path to a configuration file in TOML format.
    """
    toml_path = Path(toml_path)
    cfg = config.Config.from_toml_path(toml_path)

    if cfg.predict is None:
        raise ValueError(
            f"predict called with a config.toml file that does not have a PREDICT section: {toml_path}"
        )

    # ---- set up logging ----------------------------------------------------------------------------------------------
    config_logging_for_cli(
        log_dst=cfg.predict.output_dir,
        log_stem="predict",
        level="INFO",
        force=True,
    )
    log_version(logger)
    logger.info("Logging results to {}".format(cfg.predict.output_dir))

    if cfg.predict.dataset.path is None:
        raise ValueError(
            "No value is specified for 'dataset_path' in this .toml config file."
            f"To generate a .csv file that represents the dataset, "
            f"please run the following command:\n'vak prep {toml_path}'"
        )

    predict_module.predict(
        model_config=cfg.predict.model.asdict(),
        dataset_config=cfg.predict.dataset.asdict(),
        trainer_config=cfg.predict.trainer.asdict(),
        checkpoint_path=cfg.predict.checkpoint_path,
        labelmap_path=cfg.predict.labelmap_path,
        num_workers=cfg.predict.num_workers,
        timebins_key=(
            cfg.prep.spect_params.timebins_key
            if cfg.prep
            else common.constants.TIMEBINS_KEY
        ),
        frames_standardizer_path=cfg.predict.frames_standardizer_path,
        annot_csv_filename=cfg.predict.annot_csv_filename,
        output_dir=cfg.predict.output_dir,
        min_segment_dur=cfg.predict.min_segment_dur,
        majority_vote=cfg.predict.majority_vote,
        save_net_outputs=cfg.predict.save_net_outputs,
    )
