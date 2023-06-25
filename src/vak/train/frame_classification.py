"""Function that trains models in the frame classification family."""
from __future__ import annotations

import json
import logging
import pathlib
import shutil
import datetime

import joblib

import pandas as pd
import torch.utils.data

from .. import (
    datasets,
    models,
    transforms,
)
from ..common import validators
from ..datasets.frame_classification import (
    WindowDataset,
    FramesDataset
)
from ..common.device import get_default as get_default_device
from ..common.paths import generate_results_dir_name_as_path
from ..common.trainer import get_default_trainer


logger = logging.getLogger(__name__)


def get_split_dur(df: pd.DataFrame, split: str) -> float:
    """Get duration of a split in a dataset from a pandas DataFrame representing the dataset."""
    return df[df["split"] == split]["duration"].sum()


def train_frame_classification_model(
    model_name: str,
    model_config: dict,
    dataset_path: str | pathlib.Path,
    window_size: int,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
    checkpoint_path: str | pathlib.Path | None = None,
    spect_scaler_path: str | pathlib.Path | None = None,
    root_results_dir: str | pathlib.Path | None = None,
    results_path: str | pathlib.Path | None = None,
    normalize_spectrograms: bool = True,
    shuffle: bool = True,
    val_step: int | None = None,
    ckpt_step: int | None = None,
    patience: int | None = None,
    device: str | None = None,
    split: str = 'train',
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
    model_name : str
        Model name, must be one of vak.models.MODEL_NAMES.
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
    dataset_csv_path
        Path to csv file representing splits of dataset,
        e.g., such a file generated by running ``vak prep``.
        This parameter is used by :func:`vak.core.learncurve` to specify
        different splits to use, when generating results for a learning curve.
        If this argument is specified, the csv file must be inside the directory
        ``dataset_path``.
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
            ('checkpoint_path', 'spect_scaler_path'),
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

    logger.info(
        f"Loading dataset from path: {dataset_path}",
    )
    metadata = datasets.frame_classification.FrameClassificationDatasetMetadata.from_dataset_path(dataset_path)
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    if val_step and not dataset_df["split"].str.contains("val").any():
        raise ValueError(
            f"val_step set to {val_step} but dataset does not contain a validation set; "
            f"please run `vak prep` with a config.toml file that specifies a duration for the validation set."
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    if results_path:
        results_path = pathlib.Path(results_path).expanduser().resolve()
        if not results_path.is_dir():
            raise NotADirectoryError(
                f"results_path not recognized as a directory: {results_path}"
            )
    else:
        results_path = generate_results_dir_name_as_path(root_results_dir)
        results_path.mkdir()

    frame_dur = metadata.frame_dur
    logger.info(
        f"Duration of a frame in dataset, in seconds: {frame_dur}",
    )

    # ---------------- load training data  -----------------------------------------------------------------------------
    logger.info(f"using training dataset from {dataset_path}")
    # below, if we're going to train network to predict unlabeled segments, then
    # we need to include a class for those unlabeled segments in labelmap,
    # the mapping from labelset provided by user to a set of consecutive
    # integers that the network learns to predict
    train_dur = get_split_dur(dataset_df, "train")
    logger.info(
        f"Total duration of training split from dataset (in s): {train_dur}",
    )

    labelmap_path = dataset_path / "labelmap.json"
    logger.info(
        f"loading labelmap from path: {labelmap_path}"
    )
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)
    # copy to new results_path
    with open(results_path.joinpath("labelmap.json"), "w") as f:
        json.dump(labelmap, f)

    if spect_scaler_path is not None and normalize_spectrograms:
        logger.info(
            f"loading spect scaler from path: {spect_scaler_path}"
        )
        spect_standardizer = joblib.load(spect_scaler_path)
        shutil.copy(spect_scaler_path, results_path)
    # get transforms just before creating datasets with them
    elif normalize_spectrograms and spect_scaler_path is None:
        logger.info(
            f"no spect_scaler_path provided, not loading",
        )
        logger.info("will normalize spectrograms")
        spect_standardizer = transforms.StandardizeSpect.fit_dataset_path(
            dataset_path, split=split,
        )
        joblib.dump(spect_standardizer, results_path.joinpath("StandardizeSpect"))
    elif spect_scaler_path is not None and not normalize_spectrograms:
        raise ValueError('spect_scaler_path provided but normalize_spectrograms was False, these options conflict')
    else:
        #not normalize_spectrograms and spect_scaler_path is None:
        logger.info(
            "normalize_spectrograms is False and no spect_scaler_path was provided, "
            "will not standardize spectrograms",
            )
        spect_standardizer = None
    transform, target_transform = transforms.get_defaults("train", spect_standardizer)

    train_dataset = WindowDataset.from_dataset_path(
        dataset_path=dataset_path,
        window_size=window_size,
        split=split,
        transform=transform,
        target_transform=target_transform,
    )
    logger.info(
        f"Duration of WindowDataset used for training, in seconds: {train_dataset.duration}",
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ---------------- load validation set (if there is one) -----------------------------------------------------------
    if val_step:
        item_transform = transforms.get_defaults(
            "eval",
            spect_standardizer,
            window_size=window_size,
            return_padding_mask=True,
        )
        val_dataset = FramesDataset.from_dataset_path(
            dataset_path=dataset_path,
            split="val",
            item_transform=item_transform,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            shuffle=False,
            # batch size 1 because each spectrogram reshaped into a batch of windows
            batch_size=1,
            num_workers=num_workers,
        )
        val_dur = get_split_dur(dataset_df, "val")
        logger.info(
            f"Total duration of validation split from dataset (in s): {val_dur}",
        )

        logger.info(
            f"will measure error on validation set every {val_step} steps of training",
        )
    else:
        val_loader = None

    if device is None:
        device = get_default_device()

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
        'ckpt_root': ckpt_root,
        'ckpt_step': ckpt_step,
        'patience': patience,
    }
    trainer = get_default_trainer(
        max_steps=max_steps,
        log_save_dir=results_model_root,
        val_step=val_step,
        default_callback_kwargs=default_callback_kwargs,
        device=device,
    )
    train_time_start = datetime.datetime.now()
    logger.info(
        f"Training start time: {train_time_start.isoformat()}"
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    train_time_stop = datetime.datetime.now()
    logger.info(
        f"Training stop time: {train_time_stop.isoformat()}"
    )
    elapsed = train_time_stop - train_time_start
    logger.info(
        f"Elapsed training time: {elapsed}"
    )
