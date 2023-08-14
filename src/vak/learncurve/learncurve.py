"""High-level function that generates results for a learning curve for all models."""
from __future__ import annotations

import logging
import pathlib

from .. import models
from ..common.converters import expanded_user_path
from .frame_classification import learning_curve_for_frame_classification_model

logger = logging.getLogger(__name__)


def learning_curve(
    model_name: str,
    model_config: dict,
    dataset_path: str | pathlib.Path,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
    train_transform_params: dict | None = None,
    train_dataset_params: dict | None = None,
    val_transform_params: dict | None = None,
    val_dataset_params: dict | None = None,
    results_path: str | pathlib.Path = None,
    post_tfm_kwargs: dict | None = None,
    normalize_spectrograms: bool = True,
    shuffle: bool = True,
    val_step: int | None = None,
    ckpt_step: int | None = None,
    patience: int | None = None,
    device: str | None = None,
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
    model_name : str
        Model name, must be one of vak.models.registry.MODEL_NAMES.
    model_config : dict
        Model configuration in a ``dict``,
        as loaded from a .toml file,
        and used by the model method ``from_config``.
    dataset_path : str
        path to where dataset was saved as a csv.
    window_size : int
        size of windows taken from spectrograms, in number of time bins,
        shown to neural networks
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
    device : str
        Device on which to work with model + data.
        Default is None. If None, then a device will be selected with vak.device.get_default.
        That function defaults to 'cuda' if torch.cuda.is_available is True.
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
    """
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    dataset_path = expanded_user_path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(
            f"`dataset_path` not found or not recognized as a directory: {dataset_path}"
        )

    try:
        model_family = models.registry.MODEL_FAMILY_FROM_NAME[model_name]
    except KeyError as e:
        raise ValueError(
            f"No model family found for the model name specified: {model_name}"
        ) from e
    if model_family == "FrameClassificationModel":
        learning_curve_for_frame_classification_model(
            model_name=model_name,
            model_config=model_config,
            dataset_path=dataset_path,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_workers=num_workers,
            train_transform_params=train_transform_params,
            train_dataset_params=train_dataset_params,
            val_transform_params=val_transform_params,
            val_dataset_params=val_dataset_params,
            results_path=results_path,
            post_tfm_kwargs=post_tfm_kwargs,
            normalize_spectrograms=normalize_spectrograms,
            shuffle=shuffle,
            val_step=val_step,
            ckpt_step=ckpt_step,
            patience=patience,
            device=device,
        )
    else:
        raise ValueError(f"Model family not recognized: {model_family}")
