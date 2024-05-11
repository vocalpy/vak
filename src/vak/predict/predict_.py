"""High-level function that generates new inferences from trained models."""

from __future__ import annotations

import logging
import os
import pathlib

from .. import models
from ..common import validators
from ..common.accelerator import get_default as get_default_device
from .frame_classification import predict_with_frame_classification_model

logger = logging.getLogger(__name__)


def predict(
    model_config: dict,
    dataset_config: dict,
    trainer_config: dict,
    checkpoint_path: str | pathlib.Path,
    labelmap_path: str | pathlib.Path,
    num_workers: int = 2,
    timebins_key: str = "t",
    frames_standardizer_path: str | pathlib.Path | None = None,
    device: str | None = None,
    annot_csv_filename: str | None = None,
    output_dir: str | pathlib.Path | None = None,
    min_segment_dur: float | None = None,
    majority_vote: bool = False,
    save_net_outputs: bool = False,
):
    """Make predictions on a dataset with a trained model.

    Parameters
    ----------
    model_config : dict
        Model configuration in a ``dict``,
        as loaded from a .toml file,
        and used by the model method ``from_config``.
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
    window_size : int
        size of windows taken from spectrograms, in number of time bins,
        shown to neural networks
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader. Default is 2.
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
        If True, save 'raw' outputs of neural networks
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
        (checkpoint_path, labelmap_path, frames_standardizer_path),
        ("checkpoint_path", "labelmap_path", "frames_standardizer_path"),
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

    if device is None:
        device = get_default_device()

    model_name = model_config["name"]
    try:
        model_family = models.registry.MODEL_FAMILY_FROM_NAME[model_name]
    except KeyError as e:
        raise ValueError(
            f"No model family found for the model name specified: {model_name}"
        ) from e
    if model_family == "FrameClassificationModel":
        predict_with_frame_classification_model(
            model_config=model_config,
            dataset_config=dataset_config,
            trainer_config=trainer_config,
            checkpoint_path=checkpoint_path,
            labelmap_path=labelmap_path,
            num_workers=num_workers,
            timebins_key=timebins_key,
            frames_standardizer_path=frames_standardizer_path,
            annot_csv_filename=annot_csv_filename,
            output_dir=output_dir,
            min_segment_dur=min_segment_dur,
            majority_vote=majority_vote,
            save_net_outputs=save_net_outputs,
        )
    else:
        raise ValueError(f"Model family not recognized: {model_family}")
