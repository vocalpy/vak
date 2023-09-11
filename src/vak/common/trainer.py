from __future__ import annotations

import pathlib

import pytorch_lightning as lightning


def get_default_train_callbacks(
    ckpt_root: str | pathlib.Path,
    ckpt_step: int,
    patience: int,
):
    ckpt_callback = lightning.callbacks.ModelCheckpoint(
        dirpath=ckpt_root,
        filename="checkpoint",
        every_n_train_steps=ckpt_step,
        save_last=True,
        verbose=True,
    )
    ckpt_callback.CHECKPOINT_NAME_LAST = "checkpoint"
    ckpt_callback.FILE_EXTENSION = ".pt"

    val_ckpt_callback = lightning.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=ckpt_root,
        save_top_k=1,
        mode="max",
        filename="max-val-acc-checkpoint",
        auto_insert_metric_name=False,
        verbose=True,
    )
    val_ckpt_callback.FILE_EXTENSION = ".pt"

    early_stopping = lightning.callbacks.EarlyStopping(
        mode="max",
        monitor="val_acc",
        patience=patience,
        verbose=True,
    )

    return [ckpt_callback, val_ckpt_callback, early_stopping]


def get_default_trainer(
    max_steps: int,
    log_save_dir: str | pathlib.Path,
    val_step: int,
    default_callback_kwargs: dict | None = None,
    device: str = "cuda",
) -> lightning.Trainer:
    """Returns an instance of ``lightning.Trainer``
    with a default set of callbacks.
    Used by ``vak.core`` functions."""
    if default_callback_kwargs:
        callbacks = get_default_train_callbacks(**default_callback_kwargs)
    else:
        callbacks = None

    # TODO: use accelerator parameter, https://github.com/vocalpy/vak/issues/691
    if device == "cuda":
        accelerator = "gpu"
    else:
        accelerator = "auto"

    logger = lightning.loggers.TensorBoardLogger(save_dir=log_save_dir)

    trainer = lightning.Trainer(
        callbacks=callbacks,
        val_check_interval=val_step,
        max_steps=max_steps,
        accelerator=accelerator,
        logger=logger,
    )
    return trainer
