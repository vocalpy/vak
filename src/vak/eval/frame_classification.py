"""Function that evaluates trained models in the frame classification family."""

from __future__ import annotations

import json
import logging
import pathlib
from collections import OrderedDict
from datetime import datetime

import joblib
import lightning
import pandas as pd
import torch.utils.data

from .. import datapipes, datasets, models, transforms
from ..common import constants, validators
from ..datapipes.frame_classification import InferDatapipe

logger = logging.getLogger(__name__)


def eval_frame_classification_model(
    model_config: dict,
    dataset_config: dict,
    trainer_config: dict,
    checkpoint_path: str | pathlib.Path,
    labelmap_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    num_workers: int,
    frames_standardizer_path: str | pathlib.Path = None,
    post_tfm_kwargs: dict | None = None,
) -> None:
    """Evaluate a trained model.

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
    checkpoint_path : str, pathlib.Path
        Path to directory with checkpoint files saved by Torch, to reload model
    output_dir : str, pathlib.Path
        Path to location where .csv files with evaluation metrics should be saved.
    labelmap_path : str, pathlib.Path
        Path to 'labelmap.json' file.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader. Default is 2.
    frames_standardizer_path : str, pathlib.Path
        Path to a saved :class:`vak.transforms.FramesStandardizer`
        object used to standardize (normalize) frames.
        If frames were standardized during training and this is not provided,
        will give incorrect results. Default is None.
    post_tfm_kwargs : dict
        Keyword arguments to post-processing transform.
        If None, then no additional clean-up is applied
        when transforming labeled timebins to segments,
        the default behavior. The transform used is
        ``vak.transforms.frame_labels.PostProcess`.
        Valid keyword argument names are 'majority_vote'
        and 'min_segment_dur', and should be appropriate
        values for those arguments: Boolean for ``majority_vote``,
        a float value for ``min_segment_dur``.
        See the docstring of the transform for more details on
        these arguments and how they work.

    Notes
    -----
    Note that unlike :func:`~vak.predict.predict`, this function
    can modify ``labelmap`` so that metrics like edit distance
    are correctly computed, by converting any string labels
    in ``labelmap`` with multiple characters
    to (mock) single-character labels,
    with :func:`vak.labels.multi_char_labels_to_single_char`.
    """
    # ---- pre-conditions ----------------------------------------------------------------------------------------------
    for path, path_name in zip(
        (checkpoint_path, labelmap_path, frames_standardizer_path),
        ("checkpoint_path", "labelmap_path", "frames_standardizer_path"),
    ):
        if path is not None:  # because `frames_standardizer_path` is optional
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

    if not validators.is_a_directory(output_dir):
        raise NotADirectoryError(
            f"value for ``output_dir`` not recognized as a directory: {output_dir}"
        )

    # ---- get time for .csv file --------------------------------------------------------------------------------------
    timenow = datetime.now().strftime("%y%m%d_%H%M%S")

    # ---- load what we need to transform data -------------------------------------------------------------------------
    if frames_standardizer_path:
        logger.info(
            f"loading frames standardizer from path: {frames_standardizer_path}"
        )
        frames_standardizer = joblib.load(frames_standardizer_path)
    else:
        logger.info(
            "No `frames_standardizer_path` provided, not standardizing frames."
        )
        frames_standardizer = None

    logger.info(f"loading labelmap from path: {labelmap_path}")
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)

    # ---------------- load data for evaluation ------------------------------------------------------------------------
    if "split" in dataset_config["params"]:
        split = dataset_config["params"]["split"]
        # we do this convoluted thing to avoid 'TypeError: Dataset got multiple values for split`
        del dataset_config["params"]["split"]
    else:
        split = "test"
    # ---- *not* using a built-in dataset ------------------------------------------------------------------------------
    if dataset_config["name"] is None:
        dataset_path = pathlib.Path(dataset_config["path"])
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise NotADirectoryError(
                f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
            )

        # we unpack `frame_dur` to log it, regardless of whether we use it with post_tfm below
        metadata = datapipes.frame_classification.Metadata.from_dataset_path(
            dataset_path
        )
        frame_dur = metadata.frame_dur
        logger.info(
            f"Duration of a frame in dataset, in seconds: {frame_dur}",
        )
        val_dataset = InferDatapipe.from_dataset_path(
            dataset_path=dataset_path,
            split=split,
            window_size=dataset_config["params"]["window_size"],
            frames_standardizer=frames_standardizer,
            return_padding_mask=True,
        )
    # ---- *yes* using a built-in dataset ------------------------------------------------------------------------------
    else:
        # next line, we don't use dataset path in this code block,
        # but we need it below when we build the DataFrame with eval results.
        # we're unpacking it here just as we do above with a prep'd dataset
        dataset_path = pathlib.Path(dataset_config["path"])
        dataset_config["params"]["return_padding_mask"] = True
        val_dataset = datasets.get(
            dataset_config,
            split=split,
            frames_standardizer=frames_standardizer,
        )
        frame_dur = val_dataset.frame_dur
        logger.info(
            f"Duration of a frame in dataset, in seconds: {frame_dur}",
        )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        # batch size 1 because each spectrogram reshaped into a batch of windows
        batch_size=1,
        num_workers=num_workers,
    )

    # ---------------- do the actual evaluating ------------------------------------------------------------------------
    input_shape = val_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]

    if post_tfm_kwargs:
        post_tfm = transforms.frame_labels.PostProcess(
            timebin_dur=frame_dur,
            background_label=labelmap[constants.DEFAULT_BACKGROUND_LABEL],
            **post_tfm_kwargs,
        )
    else:
        post_tfm = None

    model = models.get(
        model_name,
        model_config,
        num_classes=len(labelmap),
        input_shape=input_shape,
        labelmap=labelmap,
        post_tfm=post_tfm,
    )

    logger.info(f"running evaluation for model: {model_name}")

    model.load_state_dict_from_path(checkpoint_path)

    trainer_logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=output_dir
    )
    trainer = lightning.pytorch.Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        logger=trainer_logger,
    )
    # TODO: check for hasattr(model, test_step) and if so run test
    # below, [0] because validate returns list of dicts, length of no. of val loaders
    metric_vals = trainer.validate(model, dataloaders=val_loader)[0]
    metric_vals = {f"avg_{k}": v for k, v in metric_vals.items()}
    for metric_name, metric_val in metric_vals.items():
        if metric_name.startswith("avg_"):
            logger.info(f"{metric_name}: {metric_val:0.5f}")

    # create a "DataFrame" with just one row which we will save as a csv;
    # the idea is to be able to concatenate csvs from multiple runs of eval
    row = OrderedDict(
        [
            ("model_name", model_name),
            ("checkpoint_path", checkpoint_path),
            ("labelmap_path", labelmap_path),
            ("frames_standardizer_path", frames_standardizer_path),
            ("dataset_path", dataset_path),
        ]
    )
    # TODO: is this still necessary after switching to Lightning? Stop saying "average"?
    # order metrics by name to be extra sure they will be consistent across runs
    row.update(
        sorted(
            [(k, v) for k, v in metric_vals.items() if k.startswith("avg_")]
        )
    )

    # pass index into dataframe, needed when using all scalar values (a single row)
    # throw away index below when saving to avoid extra column
    eval_df = pd.DataFrame(row, index=[0])
    eval_csv_path = output_dir.joinpath(f"eval_{model_name}_{timenow}.csv")
    logger.info(f"saving csv with evaluation metrics at: {eval_csv_path}")
    eval_df.to_csv(
        eval_csv_path, index=False
    )  # index is False to avoid having "Unnamed: 0" column when loading
