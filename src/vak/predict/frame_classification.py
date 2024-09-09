"""Function that generates new inferences from trained models in the frame classification family."""

from __future__ import annotations

import json
import logging
import os
import pathlib

import crowsetta
import joblib
import lightning
import numpy as np
import pandas as pd
import torch.utils.data
from attrs import define
from tqdm import tqdm

from .. import common, datapipes, datasets, models, transforms
from ..common import constants, files, validators
from ..datapipes.frame_classification import InferDatapipe

logger = logging.getLogger(__name__)


@define
class AnnotationDataFrame:
    """Data class that represents annotations
    for an audio file, in a :class:`pandas.DataFrame`.

    Used to save annotations that currently can't
    be saved with :mod:`crowsetta`, e.g. boundary times.
    """

    df: pd.DataFrame
    audio_path: str | pathlib.Path


def predict_with_frame_classification_model(
    model_config: dict,
    dataset_config: dict,
    trainer_config: dict,
    checkpoint_path: str | pathlib.Path,
    labelmap_path: str | pathlib.Path,
    num_workers: int = 2,
    timebins_key: str = "t",
    frames_standardizer_path: str | pathlib.Path | None = None,
    annot_csv_filename: str | None = None,
    output_dir: str | pathlib.Path | None = None,
    min_segment_dur: float | None = None,
    majority_vote: bool = False,
    save_net_outputs: bool = False,
    background_label: str = common.constants.DEFAULT_BACKGROUND_LABEL,
) -> None:
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
    frames_standardizer_path : str
        path to a saved :class:`vak.transforms.FramesStandardizer` object used to standardize (normalize) frames.
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
    # ---- pre-conditions ----------------------------------------------------------------------------------------------
    for path, path_name in zip(
        (checkpoint_path, labelmap_path, frames_standardizer_path),
        ("checkpoint_path", "labelmap_path", "frames_standardizer_path"),
    ):
        if path is not None:
            if not validators.is_a_file(path):
                raise FileNotFoundError(
                    f"value for ``{path_name}`` not recognized as a file: {path}"
                )

    model_name = model_config["name"]  # we use this var again below
    if "window_size" not in dataset_config["params"]:
        raise KeyError(
            f"The `dataset_config` for frame classification model '{model_name}' must include a 'params' sub-table "
            f"that sets a value for 'window_size', but received a `dataset_config` that did not:\n{dataset_config}"
        )

    dataset_path = pathlib.Path(dataset_config["path"])
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    # we do this first to make sure we can save things
    if output_dir is None:
        output_dir = pathlib.Path(os.getcwd())
    else:
        output_dir = pathlib.Path(output_dir)

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"value specified for output_dir is not recognized as a directory: {output_dir}"
        )

    # ---- load what we need to transform data -------------------------------------------------------------------------
    if frames_standardizer_path:
        logger.info(
            f"loading FramesStandardizer from path: {frames_standardizer_path}"
        )
        frames_standardizer = joblib.load(frames_standardizer_path)
    else:
        logger.info("Not loading FramesStandardizer, no path was specified")
        frames_standardizer = None

    logger.info(f"loading labelmap from path: {labelmap_path}")
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)

    # ---------------- load data for prediction ------------------------------------------------------------------------
    if "split" in dataset_config["params"]:
        split = dataset_config["params"]["split"]
        # we do this convoluted thing to avoid 'TypeError: Dataset got multiple values for split`
        del dataset_config["params"]["split"]
    else:
        split = "predict"
    # ---- *not* using a built-in dataset ------------------------------------------------------------------------------
    if dataset_config["name"] is None:
        metadata = datapipes.frame_classification.Metadata.from_dataset_path(
            dataset_path
        )
        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        metadata = (
            datapipes.frame_classification.metadata.Metadata.from_dataset_path(
                dataset_path
            )
        )
        # we use this below to convert annotations from frames to seconds
        frame_dur = metadata.frame_dur

        logger.info(
            f"loading dataset to predict from csv path: {dataset_csv_path}"
        )

        pred_dataset = InferDatapipe.from_dataset_path(
            dataset_path=dataset_path,
            split=split,
            window_size=dataset_config["params"]["window_size"],
            frames_standardizer=frames_standardizer,
            return_padding_mask=True,
        )
    # ---- *yes* using a built-in dataset ------------------------------------------------------------------------------
    else:
        # we need "target_type" below when converting predictions to annotations,
        # but fail early here if we don't have it
        if "target_type" not in dataset_config["params"]:
            from ..datasets.biosoundsegbench import VALID_TARGET_TYPES

            raise ValueError(
                "The dataset table in the configuration file requires a 'target_type' "
                "when running predictions on built-in datasets. "
                "Please add a key to the table whose value is a valid target type: "
                f"{VALID_TARGET_TYPES}"
            )
        dataset_config["params"]["return_padding_mask"] = True
        # next line, required to be true regardless of split so we set it here
        dataset_config["params"]["return_frames_path"] = True
        pred_dataset = datasets.get(
            dataset_config,
            split=split,
            frames_standardizer=frames_standardizer,
        )
        # we use this below to convert annotations from frames to seconds
        frame_dur = pred_dataset.frame_dur

    pred_loader = torch.utils.data.DataLoader(
        dataset=pred_dataset,
        shuffle=False,
        # batch size 1 because each spectrogram reshaped into a batch of windows
        batch_size=1,
        num_workers=num_workers,
    )

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

    trainer_logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=output_dir
    )
    trainer = lightning.pytorch.Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        logger=trainer_logger,
    )

    logger.info(f"running predict method of {model_name}")
    results = trainer.predict(model, pred_loader)
    # TODO: figure out how to overload `on_predict_epoch_end` to return dict
    pred_dict = {
        frames_path: y_pred
        for result in results
        for frames_path, y_pred in result.items()
    }

    # ---------------- set up to convert predictions to annotation files -----------------------------------------------
    if dataset_config["name"] is None:
        # we assume this default for now -- prep'd datasets are always multi-class frame label
        target_type = "multi_frame_labels"
    else:
        # we made sure we have this above when determining the kind of dataset
        target_type = dataset_config["params"]["target_type"]
        if isinstance(target_type, str):
            pass
        elif isinstance(target_type, (list, tuple)):
            target_type = tuple(sorted(target_type))

    if annot_csv_filename is None:
        annot_csv_filename = (
            pathlib.Path(dataset_path).stem + constants.ANNOT_CSV_SUFFIX
        )
    annot_csv_path = pathlib.Path(output_dir).joinpath(annot_csv_filename)
    logger.info(f"will save annotations in .csv file: {annot_csv_path}")

    # ----------------  converting to annotations ------------------------------------------------------------------
    progress_bar = tqdm(pred_loader)

    if dataset_config["name"] is None:
        # we're using a user-prepped dataset, not a built-in dataset
        # so assume we have metadata from above
        input_type = (
            metadata.input_type
        )  # we use this to get frame_times inside loop
        if input_type == "audio":
            audio_format = metadata.audio_format
        elif input_type == "spect":
            spect_format = metadata.spect_format
    else:
        input_type = "spect"  # assume this for now
        spect_format = common.constants.DEFAULT_SPECT_FORMAT

    annots = []
    logger.info("converting predictions to annotations")
    for ind, batch in enumerate(progress_bar):
        padding_mask, frames_path = batch["padding_mask"], batch["frames_path"]
        padding_mask = np.squeeze(padding_mask)
        if isinstance(frames_path, list) and len(frames_path) == 1:
            frames_path = frames_path[0]
        # we do all this basically to have clear naming below
        if (
            target_type == "multi_frame_labels"
            or target_type == "binary_frame_labels"
        ):
            class_logits = pred_dict[frames_path]
            boundary_logits = None
        elif target_type == "boundary_frame_labels":
            boundary_logits = pred_dict[frames_path]
            class_logits = None
        elif target_type == ("boundary_frame_labels", "multi_frame_labels"):
            class_logits, boundary_logits = pred_dict[frames_path]

        if save_net_outputs:
            # not sure if there's a better way to get outputs into right shape;
            # can't just call y_pred.reshape() because that basically flattens the whole array first
            # meaning we end up with elements in the wrong order
            # so instead we convert to sequence then stack horizontally, on column axis
            net_output = torch.hstack(class_logits.unbind())
            net_output = net_output[:, padding_mask]
            net_output = net_output.cpu().numpy()
            net_output_path = output_dir.joinpath(
                pathlib.Path(frames_path).stem
                + f"{model_name}{constants.NET_OUTPUT_SUFFIX}"
            )
            np.savez(net_output_path, net_output)

        if class_logits is not None:
            class_preds = torch.argmax(
                class_logits, dim=1
            )  # assumes class dimension is 1
            class_preds = (
                torch.flatten(class_preds).cpu().numpy()[padding_mask]
            )
        if boundary_logits is not None:
            boundary_preds = torch.argmax(
                boundary_logits, dim=1
            )  # assumes class dimension is 1
            boundary_preds = (
                torch.flatten(boundary_preds).cpu().numpy()[padding_mask]
            )

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

        # audio_fname is used for audio_path attribute of crowsetta.Annotation below
        audio_fname = files.spect.find_audio_fname(frames_path)
        if (
            target_type == "multi_frame_labels"
            or target_type == "binary_frame_labels"
        ):
            if majority_vote or min_segment_dur:
                if background_label in labelmap:
                    background_label = labelmap[background_label]
                elif (
                    "unlabeled" in labelmap
                ):  # some backward compatibility here
                    background_label = labelmap["unlabeled"]
                else:
                    background_label = 0  # set a default value anyway just to not throw an error
                class_preds = transforms.frame_labels.postprocess(
                    class_preds,
                    timebin_dur=frame_dur,
                    min_segment_dur=min_segment_dur,
                    majority_vote=majority_vote,
                    background_label=background_label,
                )
            labels, onsets_s, offsets_s = transforms.frame_labels.to_segments(
                class_preds,
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

            annot = crowsetta.Annotation(
                seq=seq,
                notated_path=audio_fname,
                annot_path=annot_csv_path.name,
            )
            annots.append(annot)

        elif target_type == "boundary_frame_labels":
            boundary_inds = (
                transforms.frame_labels.boundary_inds_from_boundary_labels(
                    boundary_preds,
                    force_boundary_first_ind=True,
                )
            )
            boundary_times = frame_times[boundary_inds]  # fancy indexing
            df = pd.DataFrame.from_records({"boundary_time": boundary_times})
            annots.append(AnnotationDataFrame(df=df, audio_path=audio_fname))
        elif target_type == ("boundary_frame_labels", "multi_frame_labels"):
            if majority_vote is False:
                logger.warn(
                    "`majority_vote` was set to False but `vak.predict.predict_with_frame_classification_model` "
                    "determined that this model predicts both multi-class labels and boundary labels, "
                    "so `majority_vote` will be set to True (to assign a single label to each segment determined by "
                    "a boundary)"
                )
            if background_label in labelmap:
                background_label = labelmap[background_label]
            elif "unlabeled" in labelmap:  # some backward compatibility here
                background_label = labelmap["unlabeled"]
            else:
                background_label = (
                    0  # set a default value anyway just to not throw an error
                )
            # Notice here we *always* call post-process, with majority_vote=True
            # because we are using boundary labels
            class_preds = transforms.frame_labels.postprocess(
                frame_labels=class_preds,
                timebin_dur=frame_dur,
                min_segment_dur=min_segment_dur,
                majority_vote=True,
                background_label=background_label,
                boundary_labels=boundary_preds,
            )
            labels, onsets_s, offsets_s = transforms.frame_labels.to_segments(
                class_preds,
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

            annot = crowsetta.Annotation(
                seq=seq,
                notated_path=audio_fname,
                annot_path=annot_csv_path.name,
            )
            annots.append(annot)
    if len(annots) < 1:
        # catch edge case where nothing was predicted
        # FIXME: this should have columns that match GenericSeq
        pd.DataFrame.from_records([]).to_csv(annot_csv_path)
    elif all([isinstance(annot, crowsetta.Annotation) for annot in annots]):
        generic_seq = crowsetta.formats.seq.GenericSeq(annots=annots)
        generic_seq.to_file(annot_path=annot_csv_path)
    elif all([isinstance(annot, AnnotationDataFrame) for annot in annots]):
        df_out = []
        for sample_num, annot_df in enumerate(annots):
            df = annot_df.df
            df["audio_path"] = str(annot_df.audio_path)
            df["sample_num"] = sample_num
            df_out.append(df)
        df_out = pd.concat(df_out)
        df_out.to_csv(annot_csv_path, index=False)
