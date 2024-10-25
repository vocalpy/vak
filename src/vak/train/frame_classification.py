"""Function that trains models in the frame classification family."""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import shutil

import lightning
import joblib
import pandas as pd
import torch.utils.data

from .. import datapipes, datasets, models, transforms
from ..common import validators
from ..datapipes.frame_classification import InferDatapipe, TrainDatapipe

logger = logging.getLogger(__name__)


def get_split_dur(df: pd.DataFrame, split: str) -> float:
    """Get duration of a split in a dataset from a pandas DataFrame representing the dataset."""
    return df[df["split"] == split]["duration"].sum()


def get_train_callbacks(
    ckpt_root: str | pathlib.Path,
    ckpt_step: int,
    patience: int,
    checkpoint_monitor: str = "val_acc",
    early_stopping_monitor: str = "val_acc",
    early_stopping_mode: str = "max",
) -> list[lightning.pytorch.callbacks.Callback]:
    ckpt_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=ckpt_root,
        filename="checkpoint",
        every_n_train_steps=ckpt_step,
        save_last=True,
        verbose=True,
    )
    ckpt_callback.CHECKPOINT_NAME_LAST = "checkpoint"
    ckpt_callback.FILE_EXTENSION = ".pt"

    val_ckpt_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor=checkpoint_monitor,
        dirpath=ckpt_root,
        save_top_k=1,
        mode="max",
        filename="max-val-acc-checkpoint",
        auto_insert_metric_name=False,
        verbose=True,
    )
    val_ckpt_callback.FILE_EXTENSION = ".pt"

    early_stopping = lightning.pytorch.callbacks.EarlyStopping(
        mode=early_stopping_mode,
        monitor=early_stopping_monitor,
        patience=patience,
        verbose=True,
    )

    return [ckpt_callback, val_ckpt_callback, early_stopping]


def get_trainer(
    accelerator: str,
    devices: int | list[int],
    max_steps: int,
    log_save_dir: str | pathlib.Path,
    val_step: int,
    callback_kwargs: dict | None = None,
) -> lightning.pytorch.Trainer:
    """Returns an instance of :class:`lightning.pytorch.Trainer`
    with a default set of callbacks.

    Used by :func:`vak.train.frame_classification`.
    The default set of callbacks is provided by
    :func:`get_default_train_callbacks`.

    Parameters
    ----------
    accelerator : str
    devices : int, list of int
    max_steps : int
    log_save_dir : str, pathlib.Path
    val_step : int
    default_callback_kwargs : dict, optional

    Returns
    -------
    trainer : lightning.pytorch.Trainer

    """
    if callback_kwargs:
        callbacks = get_train_callbacks(**callback_kwargs)
    else:
        callbacks = None

    logger = lightning.pytorch.loggers.TensorBoardLogger(save_dir=log_save_dir)

    trainer = lightning.pytorch.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        val_check_interval=val_step,
        max_steps=max_steps,
        logger=logger,
    )
    return trainer


def train_frame_classification_model(
    model_config: dict,
    dataset_config: dict,
    trainer_config: dict,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
    checkpoint_path: str | pathlib.Path | None = None,
    frames_standardizer_path: str | pathlib.Path | None = None,
    results_path: str | pathlib.Path | None = None,
    standardize_frames: bool = True,
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
    frames_standardizer_path : str, pathlib.Path
        path to a saved :class:`~vak.transforms.FramesStandardizer`
        used to standardize (normalize) frames, the input to a
        frame classification model.
        e.g., one generated by a previous run of :func:`vak.core.train`.
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
    standardize_frames : bool
        if True, use :class:`vak.transforms.FramesStandardizer` to standardize the frames.
        Normalization is done by subtracting off the mean for each row
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
        (checkpoint_path, frames_standardizer_path),
        ("checkpoint_path", "frames_standardizer_path"),
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
    # we do this first to make sure we can save things in `results_path`: copy of toml config file, labelset.json, etc
    results_path = pathlib.Path(results_path).expanduser().resolve()
    if not results_path.is_dir():
        raise NotADirectoryError(
            f"`results_path` not recognized as a directory: {results_path}"
        )
    logger.info(
        f"Will save results in `results_path`: {results_path}",
    )

    logger.info(
        f"Loading dataset from `dataset_path`: {dataset_path}\nUsing dataset config: {dataset_config}"
    )
    # ---------------- load training data  -----------------------------------------------------------------------------
    # ---- *not* using a built-in dataset ------------------------------------------------------------------------------
    if dataset_config["name"] is None:
        metadata = datapipes.frame_classification.Metadata.from_dataset_path(
            dataset_path
        )
        dataset_csv_path = dataset_path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)
        # we have to check this pre-condition here since we need `dataset_df` to check
        if val_step and not dataset_df["split"].str.contains("val").any():
            raise ValueError(
                f"val_step set to {val_step} but dataset does not contain a validation set; "
                f"please run `vak prep` with a config.toml file that specifies a duration for the validation set."
            )

        frame_dur = metadata.frame_dur
        logger.info(
            f"Duration of a frame in dataset, in seconds: {frame_dur}",
        )

        logger.info(f"Using training split from dataset: {dataset_path}")
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

        if frames_standardizer_path is not None and standardize_frames:
            logger.info(
                f"Loading frames standardizer from path: {frames_standardizer_path}"
            )
            frames_standardizer = joblib.load(frames_standardizer_path)
            shutil.copy(frames_standardizer_path, results_path)
        # get transforms just before creating datasets with them
        elif standardize_frames and frames_standardizer_path is None:
            logger.info(
                "No `frames_standardizer_path` provided, not loading",
            )
            logger.info("Will standardize (normalize) frames")
            frames_standardizer = (
                transforms.FramesStandardizer.fit_dataset_path(
                    dataset_path,
                    split="train",
                    subset=subset,
                )
            )
            joblib.dump(
                frames_standardizer,
                results_path.joinpath("FramesStandardizer"),
            )
        elif frames_standardizer_path is not None and not standardize_frames:
            raise ValueError(
                "`frames_standardizer_path` provided but `standardize_frames` was False, these options conflict"
            )
        # ---- *yes* using a built-in dataset --------------------------------------------------------------------------
        else:
            # not standardize_frames and frames_standardizer_path is None:
            logger.info(
                "`standardize_frames` is False and no `frames_standardizer_path` was provided, "
                "will not standardize spectrograms",
            )
            frames_standardizer = None

        train_dataset = TrainDatapipe.from_dataset_path(
            dataset_path=dataset_path,
            split="train",
            subset=subset,
            window_size=dataset_config["params"]["window_size"],
            frames_standardizer=frames_standardizer,
        )
    else:
        # ---- we are using a built-in dataset -----------------------------------------
        # TODO: fix this hack
        # (by doing the same thing with the built-in datapipes, making this a Boolean parameter
        # while still accepting a transform but defaulting to None)
        if "standardize_frames" not in dataset_config:
            logger.info(
                f'Adding `standardize_frames` argument to dataset_config["params"]: {standardize_frames}'
            )
            dataset_config["params"]["standardize_frames"] = standardize_frames
        train_dataset = datasets.get(
            dataset_config,
            split="train",
        )
        frame_dur = train_dataset.frame_dur
        logger.info(
            f"Duration of a frame in dataset, in seconds: {frame_dur}",
        )
        # copy labelmap from dataset to new results_path
        labelmap = train_dataset.labelmap
        with open(results_path.joinpath("labelmap.json"), "w") as fp:
            json.dump(labelmap, fp)
        frames_standardizer = getattr(
            train_dataset.item_transform, "frames_standardizer"
        )
        if frames_standardizer is not None:
            logger.info(
                "Saving `frames_standardizer` from item transform on training dataset"
            )
            joblib.dump(
                frames_standardizer,
                results_path.joinpath("FramesStandardizer"),
            )

    logger.info(
        f"Duration of {train_dataset.__class__.__name__} used for training, in seconds: {train_dataset.duration}",
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
        if dataset_config["name"] is None:
            logger.info(
                f"Using validation split from dataset:\n{dataset_path}"
            )
            val_dur = get_split_dur(dataset_df, "val")
            logger.info(
                f"Total duration of validation split from dataset (in s): {val_dur}",
            )
            val_dataset = InferDatapipe.from_dataset_path(
                dataset_path=dataset_path,
                split="val",
                **dataset_config["params"],
                frames_standardizer=frames_standardizer,
                return_padding_mask=True,
            )
        else:
            dataset_config["params"]["return_padding_mask"] = True
            val_dataset = datasets.get(
                dataset_config,
                split="val",
                frames_standardizer=frames_standardizer,
            )
        logger.info(
            f"Duration of {val_dataset.__class__.__name__} used for evaluation, in seconds: {val_dataset.duration}",
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
    if "target_type" in dataset_config["params"]:
        if isinstance(dataset_config["params"]["target_type"], list) and all([isinstance(target_type, str) for target_type in dataset_config["params"]["target_type"]]):
            multiple_targets = True
        elif isinstance(dataset_config["params"]["target_type"], str):
            multiple_targets = False
        else:
            raise ValueError(
                f'Invalid value for dataset_config["params"]["target_type"]: {dataset_config["params"]["target_type"], list}'
            )
    else:
        multiple_targets = False

    callback_kwargs = dict(
        ckpt_root=ckpt_root,
        ckpt_step=ckpt_step,
        patience=patience,
        checkpoint_monitor="val_multi_acc" if multiple_targets else "val_acc",
        early_stopping_monitor="val_multi_acc" if multiple_targets else "val_acc",
        early_stopping_mode="max",
    )
    trainer = get_trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        max_steps=max_steps,
        log_save_dir=results_model_root,
        val_step=val_step,
        callback_kwargs=callback_kwargs,
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
