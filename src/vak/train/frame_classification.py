"""Function that trains models in the frame classification family."""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import shutil

import joblib
import pandas as pd
import torch.utils.data

from .. import datapipes, models, transforms
from ..common import validators
from ..common.trainer import get_default_trainer
from ..datapipes.frame_classification import FramesDataset, TrainDatapipe

logger = logging.getLogger(__name__)


def get_split_dur(df: pd.DataFrame, split: str) -> float:
    """Get duration of a split in a dataset from a pandas DataFrame representing the dataset."""
    return df[df["split"] == split]["duration"].sum()


def train_frame_classification_model(
    model_config: dict,
    dataset_config: dict,
    trainer_config: dict,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
    checkpoint_path: str | pathlib.Path | None = None,
    spect_scaler_path: str | pathlib.Path | None = None,
    results_path: str | pathlib.Path | None = None,
    normalize_spectrograms: bool = True,
    shuffle: bool = True,
    val_step: int | None = None,
    ckpt_step: int | None = None,
    patience: int | None = None,
    subset: str | None = None,
) -> None:
    """Train a model from the frame classification family
    and save results.

    Saves checkpoint files for model,
    label map, and spectrogram scaler.
    These are saved either in ``results_path``
    if specified, or a new directory
    made inside ``root_results_dir``.

    Parameters
    ----------
    model_config : dict
        Model configuration in a :class:`dict`.
        Can be obtained by calling :meth:`vak.config.ModelConfig.asdict`.
    dataset_config: dict
        Dataset configuration in a :class:`dict`.
        Can be obtained by calling :meth:`vak.config.DatasetConfig.asdict`.
    trainer_config: dict
        Configuration for :class:`lightning.pytorch.Trainer` in a :class:`dict`.
        Can be obtained by calling :meth:`vak.config.TrainerConfig.asdict`.
    batch_size : int
        number of samples per batch presented to models during training.
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader.
        Optional, default is None.
    checkpoint_path : str, pathlib.Path
        path to a checkpoint file,
        e.g., one generated by a previous run of ``vak.core.train``.
        If specified, this checkpoint will be loaded into model.
        Used when continuing training.
        Default is None, in which case a new model is initialized.
    spect_scaler_path : str, pathlib.Path
        path to a ``SpectScaler`` used to normalize spectrograms,
        e.g., one generated by a previous run of ``vak.core.train``.
        Used when continuing training, for example on the same dataset.
        Default is None.
    root_results_dir : str, pathlib.Path
        Root directory in which a new directory will be created
        where results will be saved.
    results_path : str, pathlib.Path
        Directory where results will be saved.
        If specified, this parameter overrides ``root_results_dir``.
    shuffle: bool
        if True, shuffle training data before each epoch. Default is True.
    normalize_spectrograms : bool
        if True, use spect.utils.data.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    val_step : int
        Step on which to estimate accuracy using validation set.
        If val_step is n, then validation is carried out every time
        the global step / n is a whole number, i.e., when val_step modulo the global step is 0.
        Default is None, in which case no validation is done.
    ckpt_step : int
        Step on which to save to checkpoint file.
        If ckpt_step is n, then a checkpoint is saved every time
        the global step / n is a whole number, i.e., when ckpt_step modulo the global step is 0.
        Default is None, in which case checkpoint is only saved at the last epoch.
    patience : int
        number of validation steps to wait without performance on the
        validation set improving before stopping the training.
        Default is None, in which case training only stops after the specified number of epochs.
    subset : str
        Name of a subset from the training split of the dataset
        to use when training model. This parameter is used by
        :func:`vak.learncurve.learncurve` to specify subsets
        when training models for a learning curve.
    """
    for path, path_name in zip(
        (checkpoint_path, spect_scaler_path),
        ("checkpoint_path", "spect_scaler_path"),
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

    logger.info(
        f"Loading dataset from `dataset_path`: {dataset_path}",
    )
    metadata = datapipes.frame_classification.Metadata.from_dataset_path(
        dataset_path
    )
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    if val_step and not dataset_df["split"].str.contains("val").any():
        raise ValueError(
            f"val_step set to {val_step} but dataset does not contain a validation set; "
            f"please run `vak prep` with a config.toml file that specifies a duration for the validation set."
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    results_path = pathlib.Path(results_path).expanduser().resolve()
    if not results_path.is_dir():
        raise NotADirectoryError(
            f"results_path not recognized as a directory: {results_path}"
        )

    frame_dur = metadata.frame_dur
    logger.info(
        f"Duration of a frame in dataset, in seconds: {frame_dur}",
    )

    # ---------------- load training data  -----------------------------------------------------------------------------
    logger.info(f"Using training split from dataset: {dataset_path}")
    # below, if we're going to train network to predict unlabeled segments, then
    # we need to include a class for those unlabeled segments in labelmap,
    # the mapping from labelset provided by user to a set of consecutive
    # integers that the network learns to predict
    train_dur = get_split_dur(dataset_df, "train")
    logger.info(
        f"Total duration of training split from dataset (in s): {train_dur}",
    )

    labelmap_path = dataset_path / "labelmap.json"
    logger.info(f"loading labelmap from path: {labelmap_path}")
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)
    # copy to new results_path
    with open(results_path.joinpath("labelmap.json"), "w") as f:
        json.dump(labelmap, f)

    if spect_scaler_path is not None and normalize_spectrograms:
        logger.info(f"loading spect scaler from path: {spect_scaler_path}")
        spect_standardizer = joblib.load(spect_scaler_path)
        shutil.copy(spect_scaler_path, results_path)
    # get transforms just before creating datasets with them
    elif normalize_spectrograms and spect_scaler_path is None:
        logger.info(
            "no spect_scaler_path provided, not loading",
        )
        logger.info("will normalize spectrograms")
        spect_standardizer = transforms.StandardizeSpect.fit_dataset_path(
            dataset_path,
            split="train",
            subset=subset,
        )
        joblib.dump(
            spect_standardizer, results_path.joinpath("StandardizeSpect")
        )
    elif spect_scaler_path is not None and not normalize_spectrograms:
        raise ValueError(
            "spect_scaler_path provided but normalize_spectrograms was False, these options conflict"
        )
    else:
        # not normalize_spectrograms and spect_scaler_path is None:
        logger.info(
            "normalize_spectrograms is False and no spect_scaler_path was provided, "
            "will not standardize spectrograms",
        )
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
    transform_kwargs = {
        "spect_standardizer": spect_standardizer,
        "window_size": window_size,
    }
    train_transform = transforms.defaults.get_default_transform(
        model_name, "train", transform_kwargs=transform_kwargs
    )

    train_dataset = TrainDatapipe.from_dataset_path(
        dataset_path=dataset_path,
        split="train",
        subset=subset,
        item_transform=train_transform,
        **dataset_config["params"],
    )
    logger.info(
        f"Duration of TrainDatapipe used for training, in seconds: {train_dataset.duration}",
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ---------------- load validation set (if there is one) -----------------------------------------------------------
    if val_step:
        logger.info(
            f"Will measure error on validation set every {val_step} steps of training",
        )
        logger.info(f"Using validation split from dataset:\n{dataset_path}")
        val_dur = get_split_dur(dataset_df, "val")
        logger.info(
            f"Total duration of validation split from dataset (in s): {val_dur}",
        )

        # NOTE: we use same `transform_kwargs` here; will need to change to a `dataset_param`
        # when we factor transform *into* fixed DataPipes as above
        val_transform = transforms.defaults.get_default_transform(
            model_name, "eval", transform_kwargs
        )
        val_dataset = FramesDataset.from_dataset_path(
            dataset_path=dataset_path,
            split="val",
            item_transform=val_transform,
        )
        logger.info(
            f"Duration of FramesDataset used for evaluation, in seconds: {val_dataset.duration}",
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            shuffle=False,
            # batch size 1 because each spectrogram reshaped into a batch of windows
            batch_size=1,
            num_workers=num_workers,
        )
    else:
        val_loader = None

    model = models.get(
        model_name,
        model_config,
        num_classes=len(labelmap),
        input_shape=train_dataset.shape,
        labelmap=labelmap,
    )

    if checkpoint_path is not None:
        logger.info(
            f"loading checkpoint for {model_name} from path: {checkpoint_path}",
        )
        model.load_state_dict_from_path(checkpoint_path)

    results_model_root = results_path.joinpath(model_name)
    results_model_root.mkdir()
    ckpt_root = results_model_root.joinpath("checkpoints")
    ckpt_root.mkdir()
    logger.info(f"training {model_name}")
    max_steps = num_epochs * len(train_loader)
    default_callback_kwargs = {
        "ckpt_root": ckpt_root,
        "ckpt_step": ckpt_step,
        "patience": patience,
    }
    trainer = get_default_trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        max_steps=max_steps,
        log_save_dir=results_model_root,
        val_step=val_step,
        default_callback_kwargs=default_callback_kwargs,
    )
    train_time_start = datetime.datetime.now()
    logger.info(f"Training start time: {train_time_start.isoformat()}")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    train_time_stop = datetime.datetime.now()
    logger.info(f"Training stop time: {train_time_stop.isoformat()}")
    elapsed = train_time_stop - train_time_start
    logger.info(f"Elapsed training time: {elapsed}")
