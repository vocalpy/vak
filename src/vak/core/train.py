import os
from pathlib import Path
from datetime import datetime

import tensorflow as tf


def train_one_model(model,
                    callbacks,
                    train_loader,
                    num_epochs,
                    val_loader=None,
                    val_error_step=None,
                    ):
    model.fit_generator(train_loader,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_loader,
                        validation_freq=val_error_step,
                        class_weight=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        shuffle=True,
                        initial_epoch=0)
    model.save()


def train(models,
          optimizer,
          loss,
          metrics,
          num_epochs,
          train_loader,
          val_loader=None,
          val_error_step=None,
          checkpoint_step=None,
          patience=None,
          save_only_single_checkpoint_file=True,
          results_path=None):
    """train models using dataset specified by a csv.

    Parameters
    ----------
    models : dict
        where each key is the name of a neural network and the corresponding
        value is the configuration for that network (in a namedtuple or a dict)
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    shuffle : bool
        if True, shuffle samples in training set every epoch. Default is True.
    val_error_step : int
        step/epoch at which to estimate accuracy using validation set.
        Default is None, in which case no validation is done.
    checkpoint_step : int
        step/epoch at which to save to checkpoint file.
        Default is None, in which case checkpoint is only saved at the last epoch.
    patience : int
        number of epochs to wait without the error dropping before stopping the
        training. Default is None, in which case training continues for num_epochs
    save_only_single_checkpoint_file : bool
        if True, save only one checkpoint file instead of separate files every time
        we save. Default is True.
    results_path: str
        path to directoy where results should be saved.
        Default is None. If None, a directory is created within the current working directory.

    Returns
    -------
    None

    Saves results in root_results_dir and adds some options to config_file.
    """
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    if val_error_step and val_loader is None:
        raise ValueError(
            f"val_error_step set to {val_error_step} but no validation dataset was provided"
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    if results_path:
        results_path = Path(results_path)
    else:
        timenow = datetime.now().strftime('%y%m%d_%H%M%S')
        results_path = Path(os.getcwd()).joinpath('results_' + timenow)
        results_path.mkdir()
    if not results_path.exists():
        raise NotADirectoryError(
            f'path to directory for results not found: {results_path}'
        )

    if checkpoint_step or patience or save_only_single_checkpoint_file:
        callbacks_list = []
        if save_only_single_checkpoint_file:
            if checkpoint_step:
                period = checkpoint_step
            else:
                period = 1
            checkpointer = tf.keras.callbacks.ModelCheckpoint(results_path,
                                                           monitor='val_acc',
                                                           save_best_only=True,
                                                           period=period)
            callbacks_list.append(checkpointer)
        elif checkpoint_step:
            # logger.info(
            #     f'will save model checkpoint every {checkpoint_step} epochs'
            # )
            checkpointer = tf.keras.callbacks.ModelCheckpoint(results_path,
                                                              monitor='val_acc',
                                                              period=checkpoint_step)
            callbacks_list.append(checkpointer)

        if patience:
            # logger.info(
            #     f'will stop training model if accuracy does not go down in {patience} epochs'
            # )
            early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                                             patience=patience,
                                                             restore_best_weights=False)
            callbacks_list.append(early_stopper)
    else:
        callbacks_list = None

    for model_name, model in models.items():
        print(f'training {model_name}')
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        train_one_model(model=model,
                        callbacks=callbacks_list,
                        train_loader=train_loader,
                        num_epochs=num_epochs,
                        val_loader=val_loader,
                        val_error_step=val_error_step,
                        )
