"""Function that evaluates trained models in the parametric UMAP family."""

from __future__ import annotations

import logging
import pathlib
from collections import OrderedDict
from datetime import datetime

import lightning
import pandas as pd
import torch.utils.data

from .. import models
from ..common import validators
from ..datapipes.parametric_umap import Datapipe

logger = logging.getLogger(__name__)


def eval_parametric_umap_model(
    model_config: dict,
    dataset_config: dict,
    checkpoint_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    batch_size: int,
    num_workers: int,
    trainer_config: dict,
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
    checkpoint_path : str, pathlib.Path
        Path to directory with checkpoint files saved by Torch, to reload model
    output_dir : str, pathlib.Path
        Path to location where .csv files with evaluation metrics should be saved.
    batch_size : int
        Number of samples per batch fed into model.
    trainer_config: dict
        Configuration for :class:`lightning.pytorch.Trainer`.
        Can be obtained by calling :meth:`vak.config.TrainerConfig.asdict`.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader. Default is 2.
    split : str
        Split of dataset on which model should be evaluated.
        One of {'train', 'val', 'test'}. Default is 'test'.
    """
    # ---- pre-conditions ----------------------------------------------------------------------------------------------
    for path, path_name in zip(
        (checkpoint_path,),
        ("checkpoint_path",),
    ):
        if path is not None:  # because `frames_standardizer_path` is optional
            if not validators.is_a_file(path):
                raise FileNotFoundError(
                    f"value for ``{path_name}`` not recognized as a file: {path}"
                )

    dataset_path = pathlib.Path(dataset_config["path"])
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )
    logger.info(
        f"Loading metadata from dataset path: {dataset_path}",
    )

    if not validators.is_a_directory(output_dir):
        raise NotADirectoryError(
            f"value for ``output_dir`` not recognized as a directory: {output_dir}"
        )

    # ---- get time for .csv file --------------------------------------------------------------------------------------
    timenow = datetime.now().strftime("%y%m%d_%H%M%S")

    # ---------------- load data for evaluation ------------------------------------------------------------------------
    if "split" in dataset_config["params"]:
        split = dataset_config["params"]["split"]
    else:
        split = "test"
    model_name = model_config["name"]
    val_dataset = Datapipe.from_dataset_path(
        dataset_path=dataset_path,
        split=split,
        **dataset_config["params"],
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ---------------- do the actual evaluating ------------------------------------------------------------------------
    model = models.get(
        model_name,
        model_config,
        input_shape=val_dataset.shape,
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
    for metric_name, metric_val in metric_vals.items():
        logger.info(f"{metric_name}: {metric_val:0.5f}")

    # create a "DataFrame" with just one row which we will save as a csv;
    # the idea is to be able to concatenate csvs from multiple runs of eval
    row = OrderedDict(
        [
            ("model_name", model_name),
            ("checkpoint_path", checkpoint_path),
            ("dataset_path", dataset_path),
        ]
    )
    # order metrics by name to be extra sure they will be consistent across runs
    row.update(sorted([(k, v) for k, v in metric_vals.items()]))

    # pass index into dataframe, needed when using all scalar values (a single row)
    # throw away index below when saving to avoid extra column
    eval_df = pd.DataFrame(row, index=[0])
    eval_csv_path = output_dir.joinpath(f"eval_{model_name}_{timenow}.csv")
    logger.info(f"saving csv with evaluation metrics at: {eval_csv_path}")
    eval_df.to_csv(
        eval_csv_path, index=False
    )  # index is False to avoid having "Unnamed: 0" column when loading
