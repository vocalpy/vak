import json
from pathlib import Path

import crowsetta
import joblib
import pandas as pd
from tqdm import tqdm
import torch.utils.data

from .. import files
from .. import io
from .. import labels as labelfuncs
from ..logging import log_or_print
from .. import models
from .. import transforms
from ..datasets.unannotated_dataset import UnannotatedDataset
from ..device import get_default as get_default_device


def predict(csv_path,
            checkpoint_path,
            labelmap_path,
            annot_format,
            to_format_kwargs,
            model_config_map,
            window_size,
            num_workers=2,
            spect_key='s',
            timebins_key='t',
            spect_scaler_path=None,
            device=None,
            logger=None,
            ):
    """make predictions on dataset with trained model specified in config.toml file.
    Function called by command-line interface.

    Parameters
    ----------
    csv_path : str
        path to where dataset was saved as a csv.
    checkpoint_path : str
        path to directory with checkpoint files saved by Torch, to reload model
    labelmap_path : str
        path to 'labelmap.json' file.
    annot_format : str
        format of annotations. Any format that can be used with the
        crowsetta library is valid.
    to_format_kwargs : dict
        keyword arguments for crowsetta `to_format` function.
        Defined in .toml config file as a table.
        An example for the notmat annotation format (as a dictionary) is:
        {'min_syl_dur': 10., 'min_silent_dur', 6., 'threshold': 1500}.
    model_config_map : dict
        where each key-value pair is model name : dict of config parameters
    window_size : int
        size of windows taken from spectrograms, in number of time bins,
        shown to neural networks
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader. Default is 2.
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    device : str
        Device on which to work with model + data.
        Defaults to 'cuda' if torch.cuda.is_available is True.
    spect_scaler_path : str
        path to a saved SpectScaler object used to normalize spectrograms.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by vak.logging.get_logger. Default is None.

    Returns
    -------
    None
    """
    if device is None:
        device = get_default_device()

    # ---------------- load data for prediction ------------------------------------------------------------------------
    if spect_scaler_path:
        log_or_print(f'loading SpectScaler from path: {spect_scaler_path}', logger=logger, level='info')
        spect_standardizer = joblib.load(spect_scaler_path)
    else:
        log_or_print(f'Not loading SpectScaler, no path was specified', logger=logger, level='info')
        spect_standardizer = None

    transform, target_transform = transforms.get_defaults('predict',
                                                          spect_standardizer,
                                                          window_size=window_size,
                                                          return_padding_mask=False,
                                                          )

    log_or_print(f'loading dataset to predict from csv path: {csv_path}', logger=logger, level='info')
    pred_dataset = UnannotatedDataset.from_csv(csv_path=csv_path,
                                               split='predict',
                                               window_size=window_size,
                                               spect_key=spect_key,
                                               timebins_key=timebins_key,
                                               transform=transform,
                                               )

    pred_data = torch.utils.data.DataLoader(dataset=pred_dataset,
                                            shuffle=False,
                                            batch_size=1,  # hard coding to make this work for now
                                            num_workers=num_workers)

    # ---------------- set up to convert predictions to annotation files -----------------------------------------------
    log_or_print(f'will convert predictions to specified annotation format: {annot_format}',
                 logger=logger, level='info')
    log_or_print(f'will use following settings for converting to annotation format: {to_format_kwargs}',
                 logger=logger, level='info')
    scribe = crowsetta.Transcriber(annot_format=annot_format)
    log_or_print(f'loading labelmap from path: {labelmap_path}',
                 logger=logger, level='info')
    with labelmap_path.open('r') as f:
        labelmap = json.load(f)

    dataset_df = pd.read_csv(csv_path)
    timebin_dur = io.dataframe.validate_and_get_timebin_dur(dataset_df)
    log_or_print(f'dataset has timebins with duration: {timebin_dur}',
                 logger=logger, level='info')
    # ---------------- do the actual predicting + converting to annotations --------------------------------------------
    input_shape = pred_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]
    log_or_print(f'shape of input to networks used for predictions: {input_shape}',
                 logger=logger, level='info')

    log_or_print(f'instantiating models from model-config map:/n{model_config_map}',
                 logger=logger, level='info')
    models_map = models.from_model_config_map(
        model_config_map,
        num_classes=len(labelmap),
        input_shape=input_shape
    )
    for model_name, model in models_map.items():
        # ---------------- do the actual predicting --------------------------------------------------------------------
        log_or_print(f'loading checkpoint for {model_name} from path: {checkpoint_path}',
                     logger=logger, level='info')
        model.load(checkpoint_path)
        log_or_print(f'running predict method of {model_name}',
                     logger=logger, level='info')
        pred_dict = model.predict(pred_data=pred_data,
                                  device=device)

        # ----------------  converting to annotations ------------------------------------------------------------------
        # note use no transforms
        dataset_for_annot = UnannotatedDataset.from_csv(csv_path=csv_path,
                                                        split='predict',
                                                        window_size=window_size,
                                                        spect_key=spect_key,
                                                        timebins_key=timebins_key,
                                                        )

        data_for_annot = torch.utils.data.DataLoader(dataset=dataset_for_annot,
                                                     shuffle=False,
                                                     batch_size=1,
                                                     num_workers=num_workers)

        # use transform "outside" of Dataset so we can get back crop vec
        pad_to_window = transforms.PadToWindow(window_size,
                                               return_padding_mask=True)

        progress_bar = tqdm(data_for_annot)

        log_or_print('converting predictions to annotation files',
                     logger=logger, level='info')
        for ind, batch in enumerate(progress_bar):
            x, y = batch[0], batch[1]  # here we don't care about putting on some device outside cpu
            if len(x.shape) == 3:  # ("batch", freq_bins, time_bins)
                x = x.cpu().numpy().squeeze()
            x_pad, padding_mask = pad_to_window(x)
            y_pred_ind = pred_dict['y'].index(y)
            y_pred = pred_dict['y_pred'][y_pred_ind]
            y_pred = torch.argmax(y_pred, dim=1)  # assumes class dimension is 1
            y_pred = torch.flatten(y_pred).cpu().numpy()[padding_mask]
            labels, onsets_s, offsets_s = labelfuncs.lbl_tb2segments(y_pred,
                                                                     labelmap=labelmap,
                                                                     timebin_dur=timebin_dur)
            # DataLoader wraps strings in a tuple, need to unpack
            if type(y) == tuple and len(y) == 1:
                y = y[0]
            audio_fname = files.spect.find_audio_fname(y)
            audio_filename = Path(y).parent.joinpath(audio_fname)
            audio_filename = str(audio_filename)  # in case function doesn't accept Path
            scribe.to_format(labels=labels,
                             onsets_s=onsets_s,
                             offsets_s=offsets_s,
                             filename=audio_filename,
                             **to_format_kwargs)
