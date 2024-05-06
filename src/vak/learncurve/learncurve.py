"""High-level function that generates results for a learning curve for all models."""

from __future__ import annotations

import logging
import pathlib

from .. import models
from ..common.converters import expanded_user_path
from .frame_classification import learning_curve_for_frame_classification_model

logger = logging.getLogger(__name__)


def learning_curve(
    model_config: dict,
    dataset_config: dict,
    trainer_config: dict,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
    results_path: str | pathlib.Path = None,
    post_tfm_kwargs: dict | None = None,
    standardize_frames: bool = True,
    shuffle: bool = True,
    val_step: int | None = None,
    ckpt_step: int | None = None,
    patience: int | None = None,
) -> None:
    """Generate results for a learning curve,
    where model performance is measured as a
    function of dataset duration in seconds.

    Trains a class of model with a range of dataset durations,
    and then evaluates each trained model
    with a test set that is held constant
    (unlike the training sets).
    Results are saved in a new directory within ``root_results_dir``.

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
    batch_size : int
        number of samples per batch presented to models during training.
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader.
    results_path : str, pathlib.Path
        Directory where results will be saved.
    previous_run_path : str, pathlib.Path
        path to directory containing dataset .csv files
        that represent subsets of training set, created by
        a previous run of ``vak.core.learncurve.learning_curve``.
        Typically directory will have a name like ``results_{timestamp}``
        and the actual .csv splits will be in sub-directories with names
        corresponding to the training set duration
    post_tfm_kwargs : dict
        Keyword arguments to post-processing transform.
        If None, then no additional clean-up is applied
        when transforming labeled timebins to segments,
        the default behavior. The transform used is
        ``vak.transforms.frame_labels.ToSegmentsWithPostProcessing`.
        Valid keyword argument names are 'majority_vote'
        and 'min_segment_dur', and should be appropriate
        values for those arguments: Boolean for ``majority_vote``,
        a float value for ``min_segment_dur``.
        See the docstring of the transform for more details on
        these arguments and how they work.
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
    """
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    dataset_path = expanded_user_path(dataset_config["path"])
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    model_name = model_config["name"]
    try:
        model_family = models.registry.MODEL_FAMILY_FROM_NAME[model_name]
    except KeyError as e:
        raise ValueError(
            f"No model family found for the model name specified: {model_name}"
        ) from e
    if model_family == "FrameClassificationModel":
        learning_curve_for_frame_classification_model(
            model_config=model_config,
            dataset_config=dataset_config,
            trainer_config=trainer_config,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_workers=num_workers,
            results_path=results_path,
            post_tfm_kwargs=post_tfm_kwargs,
            standardize_frames=standardize_frames,
            shuffle=shuffle,
            val_step=val_step,
            ckpt_step=ckpt_step,
            patience=patience,
        )
    else:
        raise ValueError(f"Model family not recognized: {model_family}")
