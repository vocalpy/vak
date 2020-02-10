from functools import partial
import json
from pathlib import Path

import joblib
import torch.utils.data
from torchvision import transforms

from .. import config
from ..datasets.window_dataset import WindowDataset
from .. import models


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

    # ---------------- load data for prediction ------------------------------------------------------------------------
    if cfg.predict.spect_scaler_path:
        standardize = joblib.load(cfg.predict.spect_scaler_path)
    else:
        standardize = None

    with cfg.predict.labelmap_path.open('r') as f:
        labelmap = json.load(f)

    def to_floattensor(ndarray):
        return torch.from_numpy(ndarray).float()

    # make an "add channel" transform to use with Lambda
    # this way a spectrogram 'image' has a "channel" dimension (of size 1)
    # that convolutional layers can work on
    add_channel = partial(torch.unsqueeze, dim=1)  # add channel at first dimension because windows become batch
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

    pred_dataset = WindowDataset.from_csv(csv_path=cfg.predict.csv_path,
                                          split='predict',
                                          labelmap=labelmap,
                                          window_size=cfg.dataloader.window_size,
                                          spect_key=cfg.spect_params.spect_key,
                                          timebins_key=cfg.spect_params.timebins_key,
                                          transform=transform,
                                          target_transform=target_transform
                                          )

    pred_data = torch.utils.data.DataLoader(dataset=pred_dataset,
                                            shuffle=False,
                                            batch_size=1,
                                            num_workers=cfg.predict.num_workers)

    model_config_map = config.models.map_from_path(toml_path, cfg.train.models)
    models_map = models.from_model_config_map(
        model_config_map,
        num_classes=len(labelmap),
        input_shape=pred_dataset.shape
    )
    for model_name, model in models_map.items():
        model.predict(pred_data=pred_data,
                      device=cfg.predict.device)
