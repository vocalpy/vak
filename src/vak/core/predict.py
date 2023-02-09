import json
import logging
import os
from pathlib import Path

import crowsetta
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.utils.data

from .. import (
    constants,
    files,
    io,
    validators
)
from .. import models
from .. import transforms
from ..datasets import VocalDataset
from ..device import get_default as get_default_device


logger = logging.getLogger(__name__)


def predict(
    csv_path,
    checkpoint_path,
    labelmap_path,
    model_config_map,
    window_size,
    num_workers=2,
    spect_key="s",
    timebins_key="t",
    spect_scaler_path=None,
    device=None,
    annot_csv_filename=None,
    output_dir=None,
    min_segment_dur=None,
    majority_vote=False,
    save_net_outputs=False,
):
    """Make predictions on a dataset with a trained model.

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
    save_net_outputs : bool
         if True, save 'raw' outputs of neural networks
         before they are converted to annotations. Default is False.
         Typically the output will be "logits"
         to which a softmax transform might be applied.
         For each item in the dataset--each row in  the `csv_path` .csv--
         the output will be saved in a separate file in `output_dir`,
         with the extension `{MODEL_NAME}.output.npz`. E.g., if the input is a
         spectrogram with `spect_path` filename `gy6or6_032312_081416.npz`,
         and the network is `TweetyNet`, then the net output file
         will be `gy6or6_032312_081416.tweetynet.output.npz`.
    """
    for path, path_name in zip(
            (checkpoint_path, csv_path, labelmap_path, spect_scaler_path),
            ('checkpoint_path', 'csv_path', 'labelmap_path', 'spect_scaler_path'),
    ):
        if path is not None:
            if not validators.is_a_file(path):
                raise FileNotFoundError(
                    f"value for ``{path_name}`` not recognized as a file: {csv_path}"
                )

    if output_dir is None:
        output_dir = Path(os.getcwd())
    else:
        output_dir = Path(output_dir)

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"value specified for output_dir is not recognized as a directory: {output_dir}"
        )

    if device is None:
        device = get_default_device()

    # ---------------- load data for prediction ------------------------------------------------------------------------
    if spect_scaler_path:
        logger.info(f"loading SpectScaler from path: {spect_scaler_path}")
        spect_standardizer = joblib.load(spect_scaler_path)
    else:
        logger.info(f"Not loading SpectScaler, no path was specified")
        spect_standardizer = None

    item_transform = transforms.get_defaults(
        "predict",
        spect_standardizer,
        window_size=window_size,
        return_padding_mask=True,
    )

    logger.info(f"loading labelmap from path: {labelmap_path}")
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)

    logger.info(f"loading dataset to predict from csv path: {csv_path}")
    pred_dataset = VocalDataset.from_csv(
        csv_path=csv_path,
        split="predict",
        labelmap=labelmap,
        spect_key=spect_key,
        timebins_key=timebins_key,
        item_transform=item_transform,
    )

    pred_data = torch.utils.data.DataLoader(
        dataset=pred_dataset,
        shuffle=False,
        # batch size 1 because each spectrogram reshaped into a batch of windows
        batch_size=1,
        num_workers=num_workers,
    )

    # ---------------- set up to convert predictions to annotation files -----------------------------------------------
    if annot_csv_filename is None:
        annot_csv_filename = Path(csv_path).stem + constants.ANNOT_CSV_SUFFIX
    annot_csv_path = Path(output_dir).joinpath(annot_csv_filename)
    logger.info(f"will save annotations in .csv file: {annot_csv_path}")

    dataset_df = pd.read_csv(csv_path)
    timebin_dur = io.dataframe.validate_and_get_timebin_dur(dataset_df)
    logger.info(f"dataset has timebins with duration: {timebin_dur}")

    # ---------------- do the actual predicting + converting to annotations --------------------------------------------
    input_shape = pred_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]
    logger.info(f"shape of input to networks used for predictions: {input_shape}")

    logger.info(f"instantiating models from model-config map:/n{model_config_map}")
    models_map = models.from_model_config_map(
        model_config_map, num_classes=len(labelmap), input_shape=input_shape
    )
    for model_name, model in models_map.items():
        # ---------------- do the actual predicting --------------------------------------------------------------------
        logger.info(f"loading checkpoint for {model_name} from path: {checkpoint_path}")
        model.load(checkpoint_path, device=device)
        logger.info(f"running predict method of {model_name}")
        pred_dict = model.predict(pred_data=pred_data, device=device)

        # ----------------  converting to annotations ------------------------------------------------------------------
        progress_bar = tqdm(pred_data)

        annots = []
        logger.info("converting predictions to annotations")
        for ind, batch in enumerate(progress_bar):
            padding_mask, spect_path = batch["padding_mask"], batch["spect_path"]
            padding_mask = np.squeeze(padding_mask)
            if isinstance(spect_path, list) and len(spect_path) == 1:
                spect_path = spect_path[0]
            y_pred = pred_dict[spect_path]

            if save_net_outputs:
                # not sure if there's a better way to get outputs into right shape;
                # can't just call y_pred.reshape() because that basically flattens the whole array first
                # meaning we end up with elements in the wrong order
                # so instead we convert to sequence then stack horizontally, on column axis
                net_output = torch.hstack(y_pred.unbind())
                net_output = net_output[:, padding_mask]
                net_output = net_output.cpu().numpy()
                net_output_path = output_dir.joinpath(
                    Path(spect_path).stem + f"{model_name}{constants.NET_OUTPUT_SUFFIX}"
                )
                np.savez(net_output_path, net_output)

            y_pred = torch.argmax(y_pred, dim=1)  # assumes class dimension is 1
            y_pred = torch.flatten(y_pred).cpu().numpy()[padding_mask]

            spect_dict = files.spect.load(spect_path)
            t = spect_dict[timebins_key]

            if majority_vote or min_segment_dur:
                y_pred = transforms.labeled_timebins.postprocess(
                    y_pred,
                    timebin_dur=timebin_dur,
                    min_segment_dur=min_segment_dur,
                    majority_vote=majority_vote,
                )

            labels, onsets_s, offsets_s = transforms.labeled_timebins.to_segments(
                y_pred,
                labelmap=labelmap,
                t=t,
            )
            if labels is None and onsets_s is None and offsets_s is None:
                # handle the case when all time bins are predicted to be unlabeled
                # see https://github.com/NickleDave/vak/issues/383
                continue
            seq = crowsetta.Sequence.from_keyword(
                labels=labels, onsets_s=onsets_s, offsets_s=offsets_s
            )

            audio_fname = files.spect.find_audio_fname(spect_path)
            annot = crowsetta.Annotation(
                seq=seq, audio_path=audio_fname, annot_path=annot_csv_path.name
            )
            annots.append(annot)

        crowsetta.csv.annot2csv(annot=annots, csv_filename=annot_csv_path)
