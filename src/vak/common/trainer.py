from __future__ import annotations

import pathlib

import lightning


def get_default_train_callbacks(
    ckpt_root: str | pathlib.Path,
    ckpt_step: int,
    patience: int,
):
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
        monitor="val_acc",
        dirpath=ckpt_root,
        save_top_k=1,
        mode="max",
        filename="max-val-acc-checkpoint",
        auto_insert_metric_name=False,
        verbose=True,
    )
    val_ckpt_callback.FILE_EXTENSION = ".pt"

    early_stopping = lightning.pytorch.callbacks.EarlyStopping(
        mode="max",
        monitor="val_acc",
        patience=patience,
        verbose=True,
    )

    return [ckpt_callback, val_ckpt_callback, early_stopping]


def get_default_trainer(
    accelerator: str,
    devices: int | list[int],
    max_steps: int,
    log_save_dir: str | pathlib.Path,
    val_step: int,
    default_callback_kwargs: dict | None = None,
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
    if default_callback_kwargs:
        callbacks = get_default_train_callbacks(**default_callback_kwargs)
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
