from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .eval import eval
from .train import train
from ..io import dataframe
from .. import csv
from .. import labels
from .. import split
from ..datasets.window_dataset import WindowDataset
from ..logging import log_or_print


def learning_curve(model_config_map,
                   train_set_durs,
                   num_replicates,
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
                   shuffle=True,
                   val_step=None,
                   ckpt_step=None,
                   patience=None,
                   device=None,
                   logger=None,
                   ):
    """generate learning curve, by training models on training sets across a
    range of sizes and then measure accuracy of those models on a test set.

    Parameters
    ----------
    model_config_map : dict
        where each key-value pair is model name : dict of config parameters
    train_set_durs : list
        of int, durations in seconds of subsets taken from training data
        to create a learning curve, e.g. [5, 10, 15, 20].
    num_replicates : int
        number of times to replicate training for each training set duration
        to better estimate metrics for a training set of that size.
        Each replicate uses a different randomly drawn subset of the training
        data (but of the same duration).
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

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by vak.logging.get_logger. Default is None.

    Returns
    -------
    None

    Trains models, saves results in new directory within root_results_dir
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            f'csv_path not found: {csv_path}'
        )

    log_or_print(
        f'Using dataset from .csv: {csv_path}',
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

    log_or_print(
        f'Saving results to: {results_path}',
        logger=logger, level='info'
    )

    timebin_dur = dataframe.validate_and_get_timebin_dur(dataset_df)
    log_or_print(
        f'Size of each timebin in spectrogram, in seconds: {timebin_dur}',
        logger=logger, level='info'
    )

    # ---- subset training set -----------------------------------------------------------------------------------------
    # do all of these before training, i.e. fail early if we're going to fail
    log_or_print(
        f'Creating data sets of specified durations: {train_set_durs}',
        logger=logger, level='info'
    )

    has_unlabeled = csv.has_unlabeled(csv_path, labelset, timebins_key)
    if has_unlabeled:
        map_unlabeled = True
    else:
        map_unlabeled = False
    labelmap = labels.to_map(labelset, map_unlabeled=map_unlabeled)

    train_dur_csv_paths = defaultdict(list)
    for train_dur in train_set_durs:
        log_or_print(f'subsetting training set for training set of duration: {train_dur}',
                     logger=logger, level='info')
        results_path_this_train_dur = results_path.joinpath(f'train_dur_{train_dur}s')
        results_path_this_train_dur.mkdir()
        for replicate_num in range(1, num_replicates + 1):
            results_path_this_replicate = results_path_this_train_dur.joinpath(f'replicate_{replicate_num}')
            results_path_this_replicate.mkdir()
            # get just train split, to pass to split.dataframe
            # so we don't end up with other splits in the training set
            train_df = dataset_df[dataset_df['split'] == 'train']
            subset_df = split.dataframe(train_df, train_dur=train_dur, labelset=labelset)
            subset_df = subset_df[subset_df['split'] == 'train']  # remove rows where split was set to 'None'
            # ---- use *just* train subset to get spect vectors for WindowDataset
            (spect_id_vector,
             spect_inds_vector,
             x_inds) = WindowDataset.spect_vectors_from_df(subset_df,
                                                           window_size,
                                                           spect_key,
                                                           timebins_key,
                                                           crop_dur=train_dur,
                                                           timebin_dur=timebin_dur,
                                                           labelmap=labelmap)
            for vec_name, vec in zip(['spect_id_vector', 'spect_inds_vector', 'x_inds'],
                                     [spect_id_vector, spect_inds_vector, x_inds]):
                np.save(results_path_this_replicate.joinpath(f'{vec_name}.npy'),
                        vec)
            # keep the same validation and test set by concatenating them with the train subset
            subset_df = pd.concat(
                (subset_df,
                 dataset_df[dataset_df['split'] == 'val'],
                 dataset_df[dataset_df['split'] == 'test'],
                 )
            )

            subset_csv_name = f'{csv_path.stem}_train_dur_{train_dur}s_replicate_{replicate_num}.csv'
            subset_csv_path = results_path_this_replicate.joinpath(subset_csv_name)
            subset_df.to_csv(subset_csv_path)
            train_dur_csv_paths[train_dur].append(subset_csv_path)

    # ---- main loop that creates "learning curve" ---------------------------------------------------------------------
    log_or_print(
        f'Starting training for learning curve.',
        logger=logger, level='info'
    )
    for train_dur, csv_paths in train_dur_csv_paths.items():
        log_or_print(
            f'Training replicates for training set of size: {train_dur}s',
            logger=logger, level='info'
        )
        for replicate_num, this_train_dur_this_replicate_csv_path in enumerate(csv_paths):
            log_or_print(
                f'Training replicate {replicate_num} '
                f'using dataset from .csv file: {this_train_dur_this_replicate_csv_path}',
                logger=logger, level='info'
            )
            this_train_dur_this_replicate_results_path = this_train_dur_this_replicate_csv_path.parent
            log_or_print(
                f'Saving results to: {this_train_dur_this_replicate_results_path}',
                logger=logger, level='info'
            )

            window_dataset_kwargs = {}
            for window_dataset_kwarg in ['spect_id_vector', 'spect_inds_vector', 'x_inds']:
                window_dataset_kwargs[window_dataset_kwarg] = np.load(
                    this_train_dur_this_replicate_results_path.joinpath(f'{window_dataset_kwarg}.npy'))

            train(model_config_map,
                  this_train_dur_this_replicate_csv_path,
                  labelset,
                  window_size,
                  batch_size,
                  num_epochs,
                  num_workers,
                  results_path=this_train_dur_this_replicate_results_path,
                  spect_key=spect_key,
                  timebins_key=timebins_key,
                  normalize_spectrograms=normalize_spectrograms,
                  shuffle=shuffle,
                  val_step=val_step,
                  ckpt_step=ckpt_step,
                  patience=patience,
                  device=device,
                  logger=logger,
                  **window_dataset_kwargs
                  )

            log_or_print(
                f'Evaluating models from replicate {replicate_num} '
                f'using dataset from .csv file: {this_train_dur_this_replicate_results_path}',
                logger=logger, level='info'
            )
            for model_name in model_config_map.keys():
                log_or_print(
                    f'Evaluating model: {model_name}',
                    logger=logger, level='info'
                )
                results_model_root = this_train_dur_this_replicate_results_path.joinpath(model_name)
                ckpt_root = results_model_root.joinpath('checkpoints')
                ckpt_paths = sorted(ckpt_root.glob('*.pt'))
                if any(['max-val-acc' in str(ckpt_path) for ckpt_path in ckpt_paths]):
                    ckpt_paths = [ckpt_path for ckpt_path in ckpt_paths if 'max-val-acc' in str(ckpt_path)]
                    if len(ckpt_paths) != 1:
                        raise ValueError(
                            f'did not find a single max-val-acc checkpoint path, instead found:\n{ckpt_paths}'
                        )
                    ckpt_path = ckpt_paths[0]
                else:
                    if len(ckpt_paths) != 1:
                        raise ValueError(
                            f'did not find a single checkpoint path, instead found:\n{ckpt_paths}'
                        )
                    ckpt_path = ckpt_paths[0]
                log_or_print(
                    f'Using checkpoint: {ckpt_path}',
                    logger=logger, level='info'
                )
                labelmap_path = this_train_dur_this_replicate_results_path.joinpath('labelmap.json')
                log_or_print(
                    f'Using labelmap: {labelmap_path}',
                    logger=logger, level='info'
                )
                if normalize_spectrograms:
                    spect_scaler_path = this_train_dur_this_replicate_results_path.joinpath('StandardizeSpect')
                    log_or_print(
                        f'Using spect scaler to normalize: {spect_scaler_path}',
                        logger=logger, level='info'
                    )
                else:
                    spect_scaler_path = None

                eval(this_train_dur_this_replicate_csv_path,
                     model_config_map,
                     checkpoint_path=ckpt_path,
                     labelmap_path=labelmap_path,
                     output_dir=this_train_dur_this_replicate_results_path,
                     window_size=window_size,
                     num_workers=num_workers,
                     split='test',
                     spect_scaler_path=spect_scaler_path,
                     spect_key=spect_key,
                     timebins_key=timebins_key,
                     device=device,
                     logger=logger)

    # ---- make a csv for analysis -------------------------------------------------------------------------------------
    reg_exp_num = re.compile(r"[-+]?\d*\.\d+|\d+")  # to extract train set dur and replicate num from paths

    eval_csv_paths = sorted(results_path.glob('**/eval*.csv'))
    eval_df_0 = pd.read_csv(eval_csv_paths[0])  # use to just get columns
    eval_columns = eval_df_0.columns.tolist()  # will use below to re-order
    eval_dfs = []
    for eval_csv_path in eval_csv_paths:
        train_set_dur = reg_exp_num.findall(eval_csv_path.parents[1].name)
        if len(train_set_dur) != 1:
            raise ValueError(
                f'unable to determine training set duration from .csv path: {train_set_dur}'
            )
        else:
            train_set_dur = float(train_set_dur[0])
        replicate_num = reg_exp_num.findall(eval_csv_path.parents[0].name)
        if len(replicate_num) != 1:
            raise ValueError(
                f'unable to determine replicate number from .csv path: {train_set_dur}'
            )
        else:
            replicate_num = int(replicate_num[0])
        eval_df = pd.read_csv(eval_csv_path)
        eval_df['train_set_dur'] = train_set_dur
        eval_df['replicate_num'] = replicate_num
        eval_dfs.append(eval_df)
    all_eval_df = pd.concat(eval_dfs)
    all_eval_columns = ['train_set_dur', 'replicate_num', *eval_columns]
    all_eval_df = all_eval_df[all_eval_columns]
    all_eval_df.sort_values(by=['train_set_dur', 'replicate_num'])
    learncurve_csv_path = results_path.joinpath('learning_curve.csv')
    all_eval_df.to_csv(learncurve_csv_path, index=False)    # index=False to avoid adding "Unnamed: 0" column
