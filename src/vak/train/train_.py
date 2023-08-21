"""High-level function that trains models."""
from __future__ import annotations

import logging
import pathlib

from .. import models
from ..common import validators
from .frame_classification import train_frame_classification_model
from .parametric_umap import train_parametric_umap_model

logger = logging.getLogger(__name__)


def train(
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
    checkpoint_path: str | pathlib.Path | None = None,
    spect_scaler_path: str | pathlib.Path | None = None,
    results_path: str | pathlib.Path | None = None,
    normalize_spectrograms: bool = True,
    shuffle: bool = True,
    val_step: int | None = None,
    ckpt_step: int | None = None,
    patience: int | None = None,
    device: str | None = None,
    split: str = "train",
):
    """Train a model and save results.

    Saves checkpoint files for model,
    label map, and spectrogram scaler.
    These are saved in ``results_path``.

    Parameters
    ----------
    model_name : str
        Model name, must be one of vak.models.registry.MODEL_NAMES.
    model_config : dict
        Model configuration in a ``dict``,
        as loaded from a .toml file,
        and used by the model method ``from_config``.
    dataset_path : str
        Path to dataset, a directory generated by running ``vak prep``.
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
    train_transform_params: dict, optional
        Parameters for training data transform.
        Passed as keyword arguments.
        Optional, default is None.
    train_dataset_params: dict, optional
        Parameters for training dataset.
        Passed as keyword arguments.
        Optional, default is None.
    val_transform_params: dict, optional
        Parameters for validation data transform.
        Passed as keyword arguments.
        Optional, default is None.
    val_dataset_params: dict, optional
        Parameters for validation dataset.
        Passed as keyword arguments.
        Optional, default is None.
    checkpoint_path : str, pathlib.Path
        Path to a checkpoint file,
        e.g., one generated by a previous run of ``vak.core.train``.
        If specified, this checkpoint will be loaded into model.
        Used when continuing training.
        Default is None, in which case a new model is initialized.
    spect_scaler_path : str, pathlib.Path
        path to a ``SpectScaler`` used to normalize spectrograms,
        e.g., one generated by a previous run of ``vak.core.train``.
        Used when continuing training, for example on the same dataset.
        Default is None.
    results_path : str, pathlib.Path
        Directory where results will be saved.
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    device : str
        Device on which to work with model + data.
        Default is None. If None, then a device will be selected with vak.split.get_default.
        That function defaults to 'cuda' if torch.cuda.is_available is True.
    shuffle: bool
        if True, shuffle training data before each epoch. Default is True.
    normalize_spectrograms : bool
        if True, use spect.utils.data.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    source_ids : numpy.ndarray
        Parameter for WindowDataset. Represents the 'id' of any spectrogram,
        i.e., the index into spect_paths that will let us load it.
        Default is None.
    source_inds : numpy.ndarray
        Parameter for WindowDataset. Same length as source_ids
        but values represent indices within each spectrogram.
        Default is None.
    window_inds : numpy.ndarray
        Parameter for WindowDataset.
        Indices of each window in the dataset. The value at x[0]
        represents the start index of the first window; using that
        value, we can index into source_ids to get the path
        of the spectrogram file to load, and we can index into
        source_inds to index into the spectrogram itself
        and get the window.
        Default is None.
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
    split : str
        Name of split from dataset found at ``dataset_path`` to use
        when training model. Default is 'train'. This parameter is used by
        `vak.learncurve.learncurve` to specify specific subsets of the
        training set to use when training models for a learning curve.
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

    dataset_path = pathlib.Path(dataset_path)
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
        train_frame_classification_model(
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
            checkpoint_path=checkpoint_path,
            spect_scaler_path=spect_scaler_path,
            results_path=results_path,
            normalize_spectrograms=normalize_spectrograms,
            shuffle=shuffle,
            val_step=val_step,
            ckpt_step=ckpt_step,
            patience=patience,
            device=device,
            split=split,
        )
    elif model_family == "ParametricUMAPModel":
        train_parametric_umap_model(
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
            checkpoint_path=checkpoint_path,
            results_path=results_path,
            shuffle=shuffle,
            val_step=val_step,
            ckpt_step=ckpt_step,
            device=device,
            split=split,
        )
    else:
        raise ValueError(f"Model family not recognized: {model_family}")