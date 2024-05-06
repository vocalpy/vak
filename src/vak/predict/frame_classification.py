"""Function that generates new inferences from trained models in the frame classification family."""

from __future__ import annotations

import json
import logging
import os
import pathlib

import crowsetta
import joblib
import numpy as np
import lightning
import torch.utils.data
from tqdm import tqdm

from .. import datapipes, models, transforms
from ..common import constants, files, validators
from ..datapipes.frame_classification import FramesDataset

logger = logging.getLogger(__name__)


def predict_with_frame_classification_model(
    model_config: dict,
    dataset_config: dict,
    trainer_config: dict,
    checkpoint_path,
    labelmap_path,
    num_workers=2,
    timebins_key="t",
    spect_scaler_path=None,
    annot_csv_filename=None,
    output_dir=None,
    min_segment_dur=None,
    majority_vote=False,
    save_net_outputs=False,
):
    """Make predictions on a dataset with a trained
    :class:`~vak.models.FrameClassificationModel`.

     Parameters
     ----------
    model_config : dict
        Model configuration in a :class:`dict`.
        Can be obtained by calling :meth:`vak.config.ModelConfig.asdict`.
    dataset_config: dict
        Dataset configuration in a :class:`dict`.
        Can be obtained by calling :meth:`vak.config.DatasetConfig.asdict`.
    trainer_config: dict
        Configuration for :class:`lightning.pytorch.Trainer`.
        Can be obtained by calling :meth:`vak.config.TrainerConfig.asdict`.
    checkpoint_path : str
        path to directory with checkpoint files saved by Torch, to reload model
    labelmap_path : str
        path to 'labelmap.json' file.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader. Default is 2.
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
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
        For each item in the dataset--each row in  the `dataset_path` .csv--
        the output will be saved in a separate file in `output_dir`,
        with the extension `{MODEL_NAME}.output.npz`. E.g., if the input is a
        spectrogram with `spect_path` filename `gy6or6_032312_081416.npz`,
        and the network is `TweetyNet`, then the net output file
        will be `gy6or6_032312_081416.tweetynet.output.npz`.
    """
    for path, path_name in zip(
        (checkpoint_path, labelmap_path, spect_scaler_path),
        ("checkpoint_path", "labelmap_path", "spect_scaler_path"),
    ):
        if path is not None:
            if not validators.is_a_file(path):
                raise FileNotFoundError(
                    f"value for ``{path_name}`` not recognized as a file: {path}"
                )

    dataset_path = pathlib.Path(dataset_config["path"])
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    if output_dir is None:
        output_dir = pathlib.Path(os.getcwd())
    else:
        output_dir = pathlib.Path(output_dir)

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"value specified for output_dir is not recognized as a directory: {output_dir}"
        )

    # ---------------- load data for prediction ------------------------------------------------------------------------
    if spect_scaler_path:
        logger.info(f"loading SpectScaler from path: {spect_scaler_path}")
        spect_standardizer = joblib.load(spect_scaler_path)
    else:
        logger.info("Not loading SpectScaler, no path was specified")
        spect_standardizer = None

    model_name = model_config["name"]
    # TODO: move this into datapipe once each datapipe uses a fixed set of transforms
    # that will require adding `spect_standardizer`` as a parameter to the datapipe,
    # maybe rename to `frames_standardizer`?
    try:
        window_size = dataset_config["params"]["window_size"]
    except KeyError as e:
        raise KeyError(
            f"The `dataset_config` for frame classification model '{model_name}' must include a 'params' sub-table "
            f"that sets a value for 'window_size', but received a `dataset_config` that did not:\n{dataset_config}"
        ) from e
    transform_params = {
        "spect_standardizer": spect_standardizer,
        "window_size": window_size,
    }
    item_transform = transforms.defaults.get_default_transform(
        model_name, "predict", transform_params
    )

    logger.info(f"loading labelmap from path: {labelmap_path}")
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)

    metadata = datapipes.frame_classification.Metadata.from_dataset_path(
        dataset_path
    )
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename

    logger.info(
        f"loading dataset to predict from csv path: {dataset_csv_path}"
    )

    # TODO: fix this when we build transforms into datasets; pass in `window_size` here
    pred_dataset = FramesDataset.from_dataset_path(
        dataset_path=dataset_path,
        split="predict",
        item_transform=item_transform,
    )

    pred_loader = torch.utils.data.DataLoader(
        dataset=pred_dataset,
        shuffle=False,
        # batch size 1 because each spectrogram reshaped into a batch of windows
        batch_size=1,
        num_workers=num_workers,
    )

    # ---------------- set up to convert predictions to annotation files -----------------------------------------------
    if annot_csv_filename is None:
        annot_csv_filename = (
            pathlib.Path(dataset_path).stem + constants.ANNOT_CSV_SUFFIX
        )
    annot_csv_path = pathlib.Path(output_dir).joinpath(annot_csv_filename)
    logger.info(f"will save annotations in .csv file: {annot_csv_path}")

    metadata = (
        datapipes.frame_classification.metadata.Metadata.from_dataset_path(
            dataset_path
        )
    )
    frame_dur = metadata.frame_dur
    logger.info(
        f"Duration of a frame in dataset, in seconds: {frame_dur}",
    )

    # ---------------- do the actual predicting + converting to annotations --------------------------------------------
    input_shape = pred_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]
    logger.info(
        f"Shape of input to networks used for predictions: {input_shape}"
    )

    logger.info(f"instantiating model from config:/n{model_name}")

    model = models.get(
        model_name,
        model_config,
        num_classes=len(labelmap),
        input_shape=input_shape,
        labelmap=labelmap,
    )

    # ---------------- do the actual predicting --------------------------------------------------------------------
    logger.info(
        f"loading checkpoint for {model_name} from path: {checkpoint_path}"
    )
    model.load_state_dict_from_path(checkpoint_path)

    trainer_logger = lightning.pytorch.loggers.TensorBoardLogger(save_dir=output_dir)
    trainer = lightning.pytorch.Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        logger=trainer_logger
    )

    logger.info(f"running predict method of {model_name}")
    results = trainer.predict(model, pred_loader)
    # TODO: figure out how to overload `on_predict_epoch_end` to return dict
    pred_dict = {
        frames_path: y_pred
        for result in results
        for frames_path, y_pred in result.items()
    }
    # ----------------  converting to annotations ------------------------------------------------------------------
    progress_bar = tqdm(pred_loader)

    input_type = (
        metadata.input_type
    )  # we use this to get frame_times inside loop
    if input_type == "audio":
        audio_format = metadata.audio_format
    elif input_type == "spect":
        spect_format = metadata.spect_format
    annots = []
    logger.info("converting predictions to annotations")
    for ind, batch in enumerate(progress_bar):
        padding_mask, frames_path = batch["padding_mask"], batch["frames_path"]
        padding_mask = np.squeeze(padding_mask)
        if isinstance(frames_path, list) and len(frames_path) == 1:
            frames_path = frames_path[0]
        y_pred = pred_dict[frames_path]

        if save_net_outputs:
            # not sure if there's a better way to get outputs into right shape;
            # can't just call y_pred.reshape() because that basically flattens the whole array first
            # meaning we end up with elements in the wrong order
            # so instead we convert to sequence then stack horizontally, on column axis
            net_output = torch.hstack(y_pred.unbind())
            net_output = net_output[:, padding_mask]
            net_output = net_output.cpu().numpy()
            net_output_path = output_dir.joinpath(
                pathlib.Path(frames_path).stem
                + f"{model_name}{constants.NET_OUTPUT_SUFFIX}"
            )
            np.savez(net_output_path, net_output)

        y_pred = torch.argmax(y_pred, dim=1)  # assumes class dimension is 1
        y_pred = torch.flatten(y_pred).cpu().numpy()[padding_mask]

        if input_type == "audio":
            frames, samplefreq = constants.AUDIO_FORMAT_FUNC_MAP[audio_format](
                frames_path
            )
            frame_times = np.arange(frames.shape[-1]) / samplefreq
        elif input_type == "spect":
            spect_dict = files.spect.load(
                frames_path, spect_format=spect_format
            )
            frame_times = spect_dict[timebins_key]

        if majority_vote or min_segment_dur:
            y_pred = transforms.frame_labels.postprocess(
                y_pred,
                timebin_dur=frame_dur,
                min_segment_dur=min_segment_dur,
                majority_vote=majority_vote,
            )

        labels, onsets_s, offsets_s = transforms.frame_labels.to_segments(
            y_pred,
            labelmap=labelmap,
            frame_times=frame_times,
        )
        if labels is None and onsets_s is None and offsets_s is None:
            # handle the case when all time bins are predicted to be unlabeled
            # see https://github.com/NickleDave/vak/issues/383
            continue
        seq = crowsetta.Sequence.from_keyword(
            labels=labels, onsets_s=onsets_s, offsets_s=offsets_s
        )

        audio_fname = files.spect.find_audio_fname(frames_path)
        annot = crowsetta.Annotation(
            seq=seq, notated_path=audio_fname, annot_path=annot_csv_path.name
        )
        annots.append(annot)

    generic_seq = crowsetta.formats.seq.GenericSeq(annots=annots)
    generic_seq.to_file(annot_path=annot_csv_path)
