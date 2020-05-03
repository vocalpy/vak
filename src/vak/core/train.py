from datetime import datetime
import json
from pathlib import Path

import joblib
import pandas as pd
import torch.utils.data

from .. import csv
from .. import labels
from .. import models
from .. import summary_writer
from .. import transforms
from ..datasets.window_dataset import WindowDataset
from ..datasets.vocal_dataset import VocalDataset
from ..device import get_default as get_default_device
from ..io import dataframe
from ..logging import log_or_print


def train(model_config_map,
          csv_path,
          labelset,
          window_size,
          batch_size,
          num_epochs,
          num_workers,
          root_results_dir=None,
          results_path=None,
          spect_key='s',
          timebins_key='t',
          normalize_spectrograms=True,
          spect_id_vector=None,
          spect_inds_vector=None,
          x_inds=None,
          shuffle=True,
          val_step=None,
          ckpt_step=None,
          patience=None,
          device=None,
          logger=None,
          ):
    """train models using training set specified in config.toml file.

    Parameters
    ----------
    model_config_map : dict
        where each key-value pair is model name : dict of config parameters
    csv_path : str
        path to where dataset was saved as a csv.
    labelset : set
        of str or int, the set of labels that correspond to annotated segments
        that a network should learn to segment and classify. Note that if there
        are segments that are not annotated, e.g. silent gaps between songbird
        syllables, then `vak` will assign a dummy label to those segments
        -- you don't have to give them a label here.
    window_size : int
        size of windows taken from spectrograms, in number of time bins,
        shonw to neural networks
    batch_size : int
        number of samples per batch presented to models during training.
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    num_workers : int
        Number of processes to use for parallel loading of data.
        Argument to torch.DataLoader.
    root_results_dir : str, pathlib.Path
        Root directory in which a new directory will be created where results will be saved.
    results_path : str, pathlib.Path
        Directory where results will be saved. If specified, this parameter overrides root_results_dir.
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
    spect_id_vector : numpy.ndarray
        Parameter for WindowDataset. Represents the 'id' of any spectrogram,
        i.e., the index into spect_paths that will let us load it.
        Default is None.
    spect_inds_vector : numpy.ndarray
        Parameter for WindowDataset. Same length as spect_id_vector
        but values represent indices within each spectrogram.
        Default is None.
    x_inds : numpy.ndarray
        Parameter for WindowDataset.
        Indices of each window in the dataset. The value at x[0]
        represents the start index of the first window; using that
        value, we can index into spect_id_vector to get the path
        of the spectrogram file to load, and we can index into
        spect_inds_vector to index into the spectrogram itself
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

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by vak.logging.get_logger. Default is None.

    Returns
    -------
    None

    Trains models, saves results in new directory within root_results_dir
    """
    log_or_print(
        f'Loading dataset from .csv path: {csv_path}',
        logger=logger, level='info'
    )
    dataset_df = pd.read_csv(csv_path)
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    if val_step and not dataset_df['split'].str.contains('val').any():
        raise ValueError(
            f"val_step set to {val_step} but dataset does not contain a validation set; "
            f"please run `vak prep` with a config.toml file that specifies a duration for the validation set."
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    if results_path:
        results_path = Path(results_path).expanduser().resolve()
        if not results_path.is_dir():
            raise NotADirectoryError(
                f'results_path not recognized as a directory: {results_path}'
            )
    else:
        if root_results_dir:
            root_results_dir = Path(root_results_dir)
        else:
            root_results_dir = Path('.')
        if not root_results_dir.is_dir():
            raise NotADirectoryError(
                f'root_results_dir not recognized as a directory: {root_results_dir}'
            )
        timenow = datetime.now().strftime('%y%m%d_%H%M%S')
        results_dirname = f'results_{timenow}'
        results_path = root_results_dir.joinpath(results_dirname)
        results_path.mkdir()

    timebin_dur = dataframe.validate_and_get_timebin_dur(dataset_df)
    log_or_print(
        f'Size of timebin in spectrograms from dataset, in seconds: {timebin_dur}',
        logger=logger, level='info'
    )

    # ---------------- load training data  -----------------------------------------------------------------------------
    log_or_print(f'using training dataset from {csv_path}', logger=logger, level='info')
    # below, if we're going to train network to predict unlabeled segments, then
    # we need to include a class for those unlabeled segments in labelmap,
    # the mapping from labelset provided by user to a set of consecutive
    # integers that the network learns to predict
    train_dur = dataframe.split_dur(dataset_df, 'train')
    log_or_print(
        f'Total duration of training split from dataset (in s): {train_dur}',
        logger=logger, level='info'
    )

    has_unlabeled = csv.has_unlabeled(csv_path, labelset, timebins_key)
    if has_unlabeled:
        map_unlabeled = True
    else:
        map_unlabeled = False
    labelmap = labels.to_map(labelset, map_unlabeled=map_unlabeled)
    log_or_print(
        f'number of classes in labelmap: {len(labelmap)}',
        logger=logger, level='info'
    )
    # save labelmap in case we need it later
    with open(results_path.joinpath('labelmap.json'), 'w') as f:
        json.dump(labelmap, f)

    # get transforms just before creating datasets with them
    if normalize_spectrograms:
        # we instantiate this transform here because we want to save it
        # and don't want to add more parameters to `transforms.split.get_defaults` function
        # and make too tight a coupling between this function and that one.
        # Trade off is that this is pretty verbose (even ignoring my comments)
        log_or_print('will normalize spectrograms', logger=logger, level='info')
        spect_standardizer = transforms.StandardizeSpect.fit_df(dataset_df,
                                                                spect_key=spect_key)
        joblib.dump(spect_standardizer,
                    results_path.joinpath('StandardizeSpect'))
    else:
        spect_standardizer = None
    transform, target_transform = transforms.get_defaults('train',
                                                          spect_standardizer)

    train_dataset = WindowDataset.from_csv(csv_path=csv_path,
                                           x_inds=x_inds,
                                           spect_id_vector=spect_id_vector,
                                           spect_inds_vector=spect_inds_vector,
                                           split='train',
                                           labelmap=labelmap,
                                           window_size=window_size,
                                           spect_key=spect_key,
                                           timebins_key=timebins_key,
                                           transform=transform,
                                           target_transform=target_transform
                                           )
    log_or_print(
        f'Duration of WindowDataset used for training, in seconds: {train_dataset.duration()}',
        logger=logger, level='info'
    )
    train_data = torch.utils.data.DataLoader(dataset=train_dataset,
                                             shuffle=shuffle,
                                             batch_size=batch_size,
                                             num_workers=num_workers)

    # ---------------- load validation set (if there is one) -----------------------------------------------------------
    if val_step:
        item_transform = transforms.get_defaults('eval',
                                                 spect_standardizer,
                                                 window_size=window_size,
                                                 return_padding_mask=True,
                                                 )
        val_dataset = VocalDataset.from_csv(csv_path=csv_path,
                                            split='val',
                                            labelmap=labelmap,
                                            spect_key=spect_key,
                                            timebins_key=timebins_key,
                                            item_transform=item_transform,
                                            )
        val_data = torch.utils.data.DataLoader(dataset=val_dataset,
                                               shuffle=False,
                                               # batch size 1 because each spectrogram reshaped into a batch of windows
                                               batch_size=1,
                                               num_workers=num_workers)
        val_dur = dataframe.split_dur(dataset_df, 'val')
        log_or_print(
            f'Total duration of validation split from dataset (in s): {val_dur}',
            logger=logger, level='info'
        )

        log_or_print(
            f'will measure error on validation set every {val_step} steps of training',
            logger=logger, level='info'
        )
    else:
        val_data = None

    if device is None:
        device = get_default_device()

    models_map = models.from_model_config_map(
        model_config_map,
        num_classes=len(labelmap),
        input_shape=train_dataset.shape,
        logger=logger,
    )
    for model_name, model in models_map.items():
        results_model_root = results_path.joinpath(model_name)
        results_model_root.mkdir()
        ckpt_root = results_model_root.joinpath('checkpoints')
        ckpt_root.mkdir()
        log_or_print(f'training {model_name}', logger=logger, level='info')
        writer = summary_writer.get_summary_writer(log_dir=results_model_root,
                                                   filename_suffix=model_name)
        model.summary_writer = writer
        model.fit(train_data=train_data,
                  num_epochs=num_epochs,
                  ckpt_root=ckpt_root,
                  val_data=val_data,
                  val_step=val_step,
                  ckpt_step=ckpt_step,
                  patience=patience,
                  device=device)
