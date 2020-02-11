from functools import partial
import json
from pathlib import Path

import crowsetta
import joblib
import pandas as pd
from tqdm import tqdm
import torch.utils.data
from torchvision import transforms

from .. import config
from ..datasets.unannotated_dataset import UnannotatedDataset
from .. import io
from .. import models
from ..transforms import PadToWindow, ReshapeToWindow
from .. import util


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

    def to_floattensor(ndarray):
        return torch.from_numpy(ndarray).float()

    # make an "add channel" transform to use with Lambda
    # this way a spectrogram 'image' has a "channel" dimension (of size 1)
    # that convolutional layers can work on
    add_channel = partial(torch.unsqueeze, dim=1)  # add channel at first dimension because windows become batch
    transform = transforms.Compose(
        [transforms.Lambda(standardize),
         PadToWindow(cfg.dataloader.window_size, return_crop_vec=False),
         ReshapeToWindow(cfg.dataloader.window_size),
         transforms.Lambda(to_floattensor),
         transforms.Lambda(add_channel)]
    )

    pred_dataset = UnannotatedDataset.from_csv(csv_path=cfg.predict.csv_path,
                                               split='predict',
                                               window_size=cfg.dataloader.window_size,
                                               spect_key=cfg.spect_params.spect_key,
                                               timebins_key=cfg.spect_params.timebins_key,
                                               transform=transform,
                                               )

    pred_data = torch.utils.data.DataLoader(dataset=pred_dataset,
                                            shuffle=False,
                                            batch_size=1,  # hard coding to make this work for now
                                            num_workers=cfg.predict.num_workers)

    # ---------------- set up to convert predictions to annotation files -----------------------------------------------
    scribe = crowsetta.Transcriber(annot_format=cfg.predict.annot_format)
    with cfg.predict.labelmap_path.open('r') as f:
        labelmap = json.load(f)

    dataset_df = pd.read_csv(cfg.predict.csv_path)
    timebin_dur = io.dataframe.validate_and_get_timebin_dur(dataset_df)

    # ---------------- do the actual predicting + converting to annotations --------------------------------------------
    model_config_map = config.models.map_from_path(toml_path, cfg.predict.models)

    input_shape = pred_dataset.shape
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
        # ---------------- do the actual predicting --------------------------------------------------------------------
        model.load(cfg.predict.checkpoint_path)
        pred_dict = model.predict(pred_data=pred_data,
                                  device=cfg.predict.device)

        # ----------------  converting to annotations ------------------------------------------------------------------
        # note use no transforms
        dataset_for_annot = UnannotatedDataset.from_csv(csv_path=cfg.predict.csv_path,
                                                        split='predict',
                                                        window_size=cfg.dataloader.window_size,
                                                        spect_key=cfg.spect_params.spect_key,
                                                        timebins_key=cfg.spect_params.timebins_key,
                                                        )

        data_for_annot = torch.utils.data.DataLoader(dataset=dataset_for_annot,
                                                     shuffle=False,
                                                     batch_size=1,
                                                     num_workers=cfg.predict.num_workers)

        # use transform "outside" of Dataset so we can get back crop vec
        pad_to_window = PadToWindow(cfg.dataloader.window_size, return_crop_vec=True)

        progress_bar = tqdm(data_for_annot)

        print('converting predictions to annotation files')
        for ind, batch in enumerate(progress_bar):
            x, y = batch[0], batch[1]  # here we don't care about putting on some device outside cpu
            if len(x.shape) == 3:  # ("batch", freq_bins, time_bins)
                x = x.cpu().numpy().squeeze()
            x_pad, crop_vec = pad_to_window(x)
            y_pred_ind = pred_dict['y'].index(y)
            y_pred = pred_dict['y_pred'][y_pred_ind]
            y_pred = torch.argmax(y_pred, dim=1)  # assumes class dimension is 1
            y_pred = torch.flatten(y_pred).cpu().numpy()[crop_vec]
            labels, onsets_s, offsets_s = util.labels.lbl_tb2segments(y_pred,
                                                                      labelmap=labelmap,
                                                                      timebin_dur=timebin_dur)
            # DataLoader wraps strings in a tuple, need to unpack
            if type(y) == tuple and len(y) == 1:
                y = y[0]
            audio_fname = util.path.find_audio_fname(y)
            audio_filename = Path(y).parent.joinpath(audio_fname)
            audio_filename = str(audio_filename)  # in case function doesn't accept Path
            scribe.to_format(labels=labels,
                             onsets_s=onsets_s,
                             offsets_s=offsets_s,
                             filename=audio_filename,
                             **cfg.predict.to_format_kwargs)
