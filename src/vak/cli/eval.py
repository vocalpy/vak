from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path

import joblib
import pandas as pd
import torch.utils.data

from .. import config
from .. import models
from .. import transforms
from .. import util
from ..datasets.vocal_dataset import VocalDataset


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
    cfg = config.parse.from_toml(toml_path)

    if cfg.eval is None:
        raise ValueError(
            f'eval called with a config.toml file that does not have a EVAL section: {toml_path}'
        )

    # ---- get time for log and for .csv file --------------------------------------------------------------------------
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')

    # ---- set up logging ----------------------------------------------------------------------------------------------
    logger = util.logging.get_logger(log_dst=cfg.eval.output_dir,
                                     caller='eval',
                                     timestamp=timenow,
                                     logger_name=__name__)
    logger.info('Logging results to {}'.format(cfg.eval.output_dir))

    # ---------------- load data for evaluation ------------------------------------------------------------------------
    if cfg.eval.spect_scaler_path:
        logger.info(
            f'loading spect scaler from path: {cfg.eval.spect_scaler_path}'
        )
        spect_standardizer = joblib.load(cfg.eval.spect_scaler_path)
    else:
        logger.info(
            f'not using a spect scaler'
        )
        spect_standardizer = None

    logger.info(
        f'loading labelmap from path: {cfg.eval.spect_scaler_path}'
    )
    with cfg.eval.labelmap_path.open('r') as f:
        labelmap = json.load(f)

    item_transform = transforms.get_defaults('eval',
                                             spect_standardizer,
                                             window_size=cfg.dataloader.window_size,
                                             return_padding_mask=True,
                                             )
    logger.info(
        f'creating dataset for evaluation from: {cfg.eval.csv_path}'
    )
    val_dataset = VocalDataset.from_csv(csv_path=cfg.eval.csv_path,
                                        split='val',
                                        labelmap=labelmap,
                                        spect_key=cfg.spect_params.spect_key,
                                        timebins_key=cfg.spect_params.timebins_key,
                                        item_transform=item_transform,
                                        )
    val_data = torch.utils.data.DataLoader(dataset=val_dataset,
                                           shuffle=False,
                                           # batch size 1 because each spectrogram reshaped into a batch of windows
                                           batch_size=1,
                                           num_workers=cfg.eval.num_workers)

    # ---------------- do the actual evaluating ------------------------------------------------------------------------
    model_config_map = config.models.map_from_path(toml_path, cfg.eval.models)

    input_shape = val_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]

    models_map = models.from_model_config_map(
        model_config_map,
        num_classes=len(labelmap),
        input_shape=input_shape
    )

    for model_name, model in models_map.items():
        logger.info(
            f'running evaluation for model: {model_name}'
        )
        model.load(cfg.eval.checkpoint_path)
        metric_vals = model.evaluate(eval_data=val_data,
                                     device=cfg.eval.device)
        # create a "DataFrame" with just one row which we will save as a csv;
        # the idea is to be able to concatenate csvs from multiple runs of eval
        row = OrderedDict(
            [
                ('model_name', model_name),
                ('checkpoint_path', cfg.eval.checkpoint_path),
                ('labelmap_path', cfg.eval.labelmap_path),
                ('spect_scaler_path', cfg.eval.spect_scaler_path),
                ('csv_path', cfg.eval.csv_path),
            ]
        )
        # order metrics by name to be extra sure they will be consistent across runs
        row.update(sorted(metric_vals.items()))

        df_metrics = pd.DataFrame(row, index=[0])
        eval_csv_path = cfg.eval.output_dir.joinpath(
            f'eval_{model_name}_{timenow}.csv'
        )
        logger.info(
            f'saving csv with evaluation metrics at: {eval_csv_path}'
        )
        df_metrics.to_csv(eval_csv_path)
