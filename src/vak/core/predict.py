import json
import os
from pathlib import Path

import crowsetta
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.utils.data

from .. import files
from .. import io
from .. import labeled_timebins
from ..logging import log_or_print
from .. import models
from .. import transforms
from ..datasets import VocalDataset
from ..device import get_default as get_default_device


def predict(csv_path,
            checkpoint_path,
            labelmap_path,
            model_config_map,
            window_size,
            num_workers=2,
            spect_key='s',
            timebins_key='t',
            spect_scaler_path=None,
            device=None,
            annot_csv_filename=None,
            output_dir=None,
            min_segment_dur=None,
            majority_vote=False,
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
    annot_csv_filename : str
        name of .csv file containing predicted annotations.
        Default is None, in which case the name of the dataset .csv
        is used, with '.annot.csv' appended to it.
    output_dir : str, Path
        path to location where .csv containing predicted annotation
        should be saved. Defaults to current working directory.
    min_segment_dur : float
        minimum duration of segment, in seconds. If specified, then
        any segment with a duration less than min_segment_dur is
        removed from lbl_tb. Default is None, in which case no
        segments are removed.
    majority_vote : bool
        if True, transform segments containing multiple labels
        into segments with a single label by taking a "majority vote",
        i.e. assign all time bins in the segment the most frequently
        occurring label in the segment. This transform can only be
        applied if the labelmap contains an 'unlabeled' label,
        because unlabeled segments makes it possible to identify
        the labeled segments. Default is False.

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by vak.logging.get_logger. Default is None.

    Returns
    -------
    None
    """
    if output_dir is None:
        output_dir = Path(os.getcwd())
    else:
        output_dir = Path(output_dir)

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f'value specified for output_dir is not recognized as a directory: {output_dir}'
        )

    if device is None:
        device = get_default_device()

    # ---------------- load data for prediction ------------------------------------------------------------------------
    if spect_scaler_path:
        log_or_print(f'loading SpectScaler from path: {spect_scaler_path}', logger=logger, level='info')
        spect_standardizer = joblib.load(spect_scaler_path)
    else:
        log_or_print(f'Not loading SpectScaler, no path was specified', logger=logger, level='info')
        spect_standardizer = None

    item_transform = transforms.get_defaults('predict',
                                             spect_standardizer,
                                             window_size=window_size,
                                             return_padding_mask=True,
                                             )

    log_or_print(f'loading labelmap from path: {labelmap_path}',
                 logger=logger, level='info')
    with labelmap_path.open('r') as f:
        labelmap = json.load(f)

    log_or_print(f'loading dataset to predict from csv path: {csv_path}', logger=logger, level='info')
    pred_dataset = VocalDataset.from_csv(csv_path=csv_path,
                                         split='predict',
                                         labelmap=labelmap,
                                         spect_key=spect_key,
                                         timebins_key=timebins_key,
                                         item_transform=item_transform,
                                         )

    pred_data = torch.utils.data.DataLoader(dataset=pred_dataset,
                                            shuffle=False,
                                            # batch size 1 because each spectrogram reshaped into a batch of windows
                                            batch_size=1,
                                            num_workers=num_workers)

    # ---------------- set up to convert predictions to annotation files -----------------------------------------------
    if annot_csv_filename is None:
        annot_csv_filename = Path(csv_path).stem + '.annot.csv'
    annot_csv_path = Path(output_dir).joinpath(annot_csv_filename)
    log_or_print(f'will save annotations in .csv file: {annot_csv_path}',
                 logger=logger, level='info')

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
        progress_bar = tqdm(pred_data)

        annots = []
        log_or_print('converting predictions to annotations',
                     logger=logger, level='info')
        for ind, batch in enumerate(progress_bar):
            padding_mask, spect_path = batch['padding_mask'], batch['spect_path']
            padding_mask = np.squeeze(padding_mask)
            if isinstance(spect_path, list) and len(spect_path) == 1:
                spect_path = spect_path[0]
            y_pred = pred_dict[spect_path]
            y_pred = torch.argmax(y_pred, dim=1)  # assumes class dimension is 1
            y_pred = torch.flatten(y_pred).cpu().numpy()[padding_mask]

            spect_dict = files.spect.load(spect_path)
            t = spect_dict[timebins_key]
            labels, onsets_s, offsets_s = labeled_timebins.lbl_tb2segments(y_pred,
                                                                           labelmap=labelmap,
                                                                           t=t,
                                                                           min_segment_dur=min_segment_dur,
                                                                           majority_vote=majority_vote)
            seq = crowsetta.Sequence.from_keyword(labels=labels,
                                                  onsets_s=onsets_s,
                                                  offsets_s=offsets_s)

            audio_fname = files.spect.find_audio_fname(spect_path)
            annot = crowsetta.Annotation(seq=seq, audio_path=audio_fname, annot_path=annot_csv_path.name)
            annots.append(annot)

        crowsetta.csv.annot2csv(annot=annots,
                                csv_filename=annot_csv_path)
