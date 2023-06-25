"""Function that evaluates trained models in the frame classification family."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import json
import logging
import pathlib

import joblib
import pytorch_lightning as lightning
import pandas as pd
import torch.utils.data

from .. import (
    datasets,
    models,
    transforms,
)
from ..common import validators
from ..datasets.frame_classification import FramesDataset


logger = logging.getLogger(__name__)


def eval_frame_classification_model(
    model_name: str,
    model_config: dict,
    dataset_path: str | pathlib.Path,
    checkpoint_path: str | pathlib.Path,
    labelmap_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    window_size: int,
    num_workers: int,
    split: str = "test",
    spect_scaler_path: str | pathlib.Path = None,
    post_tfm_kwargs: dict | None = None,
    device: str | None = None,
) -> None:
    """Evaluate a trained model.

    Parameters
    ----------
    model_name : str
        Model name, must be one of vak.models.MODEL_NAMES.
    model_config : dict
        Model configuration in a ``dict``,
        as loaded from a .toml file,
        and used by the model method ``from_config``.
    dataset_path : str, pathlib.Path
        Path to dataset, e.g., a csv file generated by running ``vak prep``.
    checkpoint_path : str, pathlib.Path
        path to directory with checkpoint files saved by Torch, to reload model
    output_dir : str, pathlib.Path
        Path to location where .csv files with evaluation metrics should be saved.
    window_size : int
        size of windows taken from spectrograms, in number of time bins,
        shown to neural networks
    labelmap_path : str, pathlib.Path
        path to 'labelmap.json' file.
    models : list
        of model names. e.g., 'models = TweetyNet, GRUNet, ConvNet'
    batch_size : int
        number of samples per batch presented to models during training.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader. Default is 2.
    split : str
        split of dataset on which model should be evaluated.
        One of {'train', 'val', 'test'}. Default is 'test'.
    spect_scaler_path : str, pathlib.Path
        path to a saved SpectScaler object used to normalize spectrograms.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.
        Default is None.
    post_tfm_kwargs : dict
        Keyword arguments to post-processing transform.
        If None, then no additional clean-up is applied
        when transforming labeled timebins to segments,
        the default behavior. The transform used is
        ``vak.transforms.labeled_timebins.PostProcess`.
        Valid keyword argument names are 'majority_vote'
        and 'min_segment_dur', and should be appropriate
        values for those arguments: Boolean for ``majority_vote``,
        a float value for ``min_segment_dur``.
        See the docstring of the transform for more details on
        these arguments and how they work.
    device : str
        Device on which to work with model + data.
        Defaults to 'cuda' if torch.cuda.is_available is True.

    Notes
    -----
    Note that unlike ``core.predict``, this function
    can modify ``labelmap`` so that metrics like edit distance
    are correctly computed, by converting any string labels
    in ``labelmap`` with multiple characters
    to (mock) single-character labels,
    with ``vak.labels.multi_char_labels_to_single_char``.
    """
    # ---- pre-conditions ----------------------------------------------------------------------------------------------
    for path, path_name in zip(
            (checkpoint_path, labelmap_path, spect_scaler_path),
            ('checkpoint_path', 'labelmap_path', 'spect_scaler_path'),
    ):
        if path is not None:  # because `spect_scaler_path` is optional
            if not validators.is_a_file(path):
                raise FileNotFoundError(
                    f"value for ``{path_name}`` not recognized as a file: {path}"
                )

    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    # we unpack `frame_dur` to log it, regardless of whether we use it with post_tfm below
    metadata = datasets.frame_classification.FrameClassificationDatasetMetadata.from_dataset_path(dataset_path)
    frame_dur = metadata.frame_dur
    logger.info(
        f"Duration of a frame in dataset, in seconds: {frame_dur}",
    )

    if not validators.is_a_directory(output_dir):
        raise NotADirectoryError(
            f'value for ``output_dir`` not recognized as a directory: {output_dir}'
        )

    # ---- get time for .csv file --------------------------------------------------------------------------------------
    timenow = datetime.now().strftime("%y%m%d_%H%M%S")

    # ---------------- load data for evaluation ------------------------------------------------------------------------
    if spect_scaler_path:
        logger.info(f"loading spect scaler from path: {spect_scaler_path}")
        spect_standardizer = joblib.load(spect_scaler_path)
    else:
        logger.info(f"not using a spect scaler")
        spect_standardizer = None

    logger.info(f"loading labelmap from path: {labelmap_path}")
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)

    item_transform = transforms.get_defaults(
        "eval",
        spect_standardizer,
        window_size=window_size,
        return_padding_mask=True,
    )

    val_dataset = FramesDataset.from_dataset_path(
        dataset_path=dataset_path,
        split=split,
        item_transform=item_transform,
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
        post_tfm = transforms.labeled_timebins.PostProcess(
            timebin_dur=frame_dur,
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

    if device == 'cuda':
        accelerator = 'gpu'
    else:
        accelerator = None

    trainer_logger = lightning.loggers.TensorBoardLogger(
        save_dir=output_dir
    )
    trainer = lightning.Trainer(accelerator=accelerator, logger=trainer_logger)
    # TODO: check for hasattr(model, test_step) and if so run test
    # below, [0] because validate returns list of dicts, length of no. of val loaders
    metric_vals = trainer.validate(model, dataloaders=val_loader)[0]
    metric_vals = {f'avg_{k}': v for k, v in metric_vals.items()}
    for metric_name, metric_val in metric_vals.items():
        if metric_name.startswith('avg_'):
            logger.info(
                f'{metric_name}: {metric_val:0.5f}'
            )

    # create a "DataFrame" with just one row which we will save as a csv;
    # the idea is to be able to concatenate csvs from multiple runs of eval
    row = OrderedDict(
        [
            ("model_name", model_name),
            ("checkpoint_path", checkpoint_path),
            ("labelmap_path", labelmap_path),
            ("spect_scaler_path", spect_scaler_path),
            ("dataset_path", dataset_path),
        ]
    )
    # TODO: is this still necessary after switching to Lightning? Stop saying "average"?
    # order metrics by name to be extra sure they will be consistent across runs
    row.update(
        sorted([(k, v) for k, v in metric_vals.items() if k.startswith("avg_")])
    )

    # pass index into dataframe, needed when using all scalar values (a single row)
    # throw away index below when saving to avoid extra column
    eval_df = pd.DataFrame(row, index=[0])
    eval_csv_path = output_dir.joinpath(f"eval_{model_name}_{timenow}.csv")
    logger.info(f"saving csv with evaluation metrics at: {eval_csv_path}")
    eval_df.to_csv(
        eval_csv_path, index=False
    )  # index is False to avoid having "Unnamed: 0" column when loading
