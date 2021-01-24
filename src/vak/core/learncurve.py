from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pprint
import re
import shutil

import numpy as np
import pandas as pd

from .eval import eval
from .train import train
from ..io import dataframe
from .. import csv
from .. import labels
from .. import split
from ..converters import expanded_user_path
from ..datasets.window_dataset import WindowDataset
from ..logging import log_or_print


# pattern used by path.glob to find all training data subset csvs within previous_run_path
CSV_GLOB = '**/*prep*csv'
# pattern used by re to get training subset duration
TRAIN_DUR_PAT = r'train_dur_(\d+\.\d+|\d+)s'


def train_dur_csv_paths_from_dir(previous_run_path,
                                 train_set_durs,
                                 num_replicates,
                                 results_path,
                                 logger=None):
    """return a ``dict`` mapping training dataset durations to dataset csv paths
    from a previous run of `vak.core.learncurve.learning_curve`.

    Recovers the previous subsets of the total training set, so that
    they can be used to re-run a different experiment on the same subsets for comparison.

    Parameters
    ----------
    previous_run_path : str, Path
        path to directory containing dataset .csv files
        that represent subsets of training set, created by
        a previous run of ``vak.core.learncurve.learning_curve``.
        Typically directory will have a name like ``results_{timestamp}``
        and the actual .csv splits will be in sub-directories with names
        corresponding to the training set duration
    train_set_durs : list
        of int, durations in seconds of subsets taken from training data
        to create a learning curve, e.g. [5, 10, 15, 20].
    num_replicates : int
        number of times to replicate training for each training set duration
        to better estimate metrics for a training set of that size.
        Each replicate uses a different randomly drawn subset of the training
        data (but of the same duration).
    results_path : str, pathlib.Path
        Directory where results will be saved, including
        files representing subsets of training data that this function makes.
        Path derived from the ``root_results_dir`` argument
         to ``vak.core.learncurve.learning_curve``, unless specified by user.

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by ``vak.logging.get_logger``. Default is None.

    Returns
    -------
    train_dur_csv_paths : dict
        where keys are duration in seconds of subsets taken from training data,
        and corresponding values are lists of paths to .csv files containing
        those subsets

    Notes
    -----
    ``train_set_durs`` and ``num_replicates`` are used to verify that
    the correct number of training data subsets are found for each specified
    duration, i.e., that the new experiment will correctly use
    all datasets from a previous run
    """
    previous_run_path = Path(previous_run_path)
    csv_paths = sorted(previous_run_path.glob(CSV_GLOB))

    train_dur_csv_paths = defaultdict(list)
    for train_subset_path in csv_paths:
        train_dur = re.findall(TRAIN_DUR_PAT, train_subset_path.name)
        if len(train_dur) != 1:
            raise ValueError(
                f'did not find just a single training subset duration in filename:\n'
                f'{train_subset_path}\n'
                f'Instead found: {train_dur}'
            )
        train_dur = int(train_dur[0])
        train_dur_csv_paths[train_dur].append(train_subset_path)

    found_train_set_durs = sorted(train_dur_csv_paths.keys())
    if not found_train_set_durs == sorted(train_set_durs):
        raise ValueError(
            f'training set durations found in {previous_run_path} did not match '
            f'specified training set durations.\n'
            f'Specified: {sorted(train_set_durs)}\n'
            f'Found: {found_train_set_durs}'
        )

    if not all(
            [len(csv_paths) == num_replicates
             for csv_paths in train_dur_csv_paths.values()]
    ):
        raise ValueError(
            'did not find correct number of replicates for all training set durations.'
            f'Found the following:\n{pprint(train_dur_csv_paths, indent=4)}'
        )

    log_or_print(
        f'Using the following training subsets from previous run path:\n'
        f'{pprint(train_dur_csv_paths, indent=4)}',
        logger=logger, level='info'
    )

    # need to copy .csv files, and change path in train_dur_csv_paths to point to copies
    # so that `vak.train` doesn't try to write over existing results dir, causing a crash
    for train_dur in train_dur_csv_paths.keys():  # use keys so we can modify dict inside loop
        results_path_this_train_dur = results_path.joinpath(f'train_dur_{train_dur}s')
        results_path_this_train_dur.mkdir()
        csv_paths = train_dur_csv_paths[train_dur]
        new_csv_paths = []
        for replicate_num, csv_path in zip(range(1, len(csv_paths) + 1), csv_paths):
            results_path_this_replicate = results_path_this_train_dur.joinpath(f'replicate_{replicate_num}')
            results_path_this_replicate.mkdir()
            # copy csv using Path.rename() method, append returned new path to list
            new_csv_paths.append(
                shutil.copy(src=csv_path,
                            dst=results_path_this_replicate.joinpath(csv_path.name))
            )
            # also need to copy the spect_id_vectors, etc. used with WindowDataset
            for vec_name in ('spect_id_vector', 'spect_inds_vector', 'x_inds'):
                src_vec_path = csv_path.parent.joinpath(f'{vec_name}.npy')
                shutil.copy(
                    src=src_vec_path,
                    dst=results_path_this_replicate.joinpath(src_vec_path.name)
                )

        train_dur_csv_paths[train_dur] = new_csv_paths

    return train_dur_csv_paths


def train_dur_csv_paths_from_df(dataset_df,
                                csv_path,
                                train_set_durs,
                                timebin_dur,
                                num_replicates,
                                results_path,
                                labelset,
                                window_size,
                                spect_key,
                                timebins_key,
                                labelmap,
                                logger=None):
    """return a ``dict`` mapping training dataset durations to dataset csv paths.

    csv paths representing subsets of the training data are generated using ``vak.split``.

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        representing an entire dataset of vocalizations.
    csv_path : str
        path to where dataset was saved as a csv.
    train_set_durs : list
        of int, durations in seconds of subsets taken from training data
        to create a learning curve, e.g. [5, 10, 15, 20].
    timebin_dur : float
        duration of timebins in spectrograms
    num_replicates : int
        number of times to replicate training for each training set duration
        to better estimate metrics for a training set of that size.
        Each replicate uses a different randomly drawn subset of the training
        data (but of the same duration).
    results_path : str, pathlib.Path
        Directory where results will be saved, including
        files representing subsets of training data that this function makes.
        Path derived from the ``root_results_dir`` argument
         to ``vak.core.learncurve.learning_curve``, unless specified by user.
    labelset : set
        of str or int, the set of labels that correspond to annotated segments
        that a network should learn to segment and classify. Note that if there
        are segments that are not annotated, e.g. silent gaps between songbird
        syllables, then `vak` will assign a dummy label to those segments
        -- you don't have to give them a label here.
    window_size : int
        size of windows taken from spectrograms, in number of time bins,
        shonw to neural networks
    spect_key : str
        key for accessing spectrogram in files. Default is 's'.
    timebins_key : str
        key for accessing vector of time bins in files. Default is 't'.
    labelmap : dict
        that maps labelset to consecutive integers

    Other Parameters
    ----------------
    logger : logging.Logger
        instance created by ``vak.logging.get_logger``. Default is None.

    Returns
    -------
    train_dur_csv_paths : dict
        where keys are duration in seconds of subsets taken from training data,
        and corresponding values are lists of paths to .csv files containing
        those subsets
    """
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
    return train_dur_csv_paths


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
                   previous_run_path=None,
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
    previous_run_path : str, Path
        path to directory containing dataset .csv files
        that represent subsets of training set, created by
        a previous run of ``vak.core.learncurve.learning_curve``.
        Typically directory will have a name like ``results_{timestamp}``
        and the actual .csv splits will be in sub-directories with names
        corresponding to the training set duration
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
        instance created by ``vak.logging.get_logger``. Default is None.

    Returns
    -------
    None

    Trains models, saves results in new directory within root_results_dir
    """
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    csv_path = expanded_user_path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f'csv_path not found: {csv_path}'
        )

    log_or_print(
        f'Using dataset from .csv: {csv_path}',
        logger=logger, level='info'
    )
    dataset_df = pd.read_csv(csv_path)

    if previous_run_path:
        previous_run_path = expanded_user_path(previous_run_path)
        if not previous_run_path.is_dir():
            raise NotADirectoryError(
                f'previous_run_path not recognized as a directory:\n{previous_run_path}'
            )

    if val_step and not dataset_df['split'].str.contains('val').any():
        raise ValueError(
            f"val_step set to {val_step} but dataset does not contain a validation set; "
            f"please run `vak prep` with a config.toml file that specifies a duration for the validation set."
        )

    # ---- set up directory to save output -----------------------------------------------------------------------------
    if results_path:
        results_path = expanded_user_path(results_path)
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

    # ---- get training set subsets ------------------------------------------------------------------------------------
    has_unlabeled = csv.has_unlabeled(csv_path, labelset, timebins_key)
    if has_unlabeled:
        map_unlabeled = True
    else:
        map_unlabeled = False
    labelmap = labels.to_map(labelset, map_unlabeled=map_unlabeled)

    if previous_run_path:
        log_or_print(
            f'Loading previous training subsets from:\n{previous_run_path}',
            logger=logger, level='info'
        )
        train_dur_csv_paths = train_dur_csv_paths_from_dir(previous_run_path,
                                                           train_set_durs,
                                                           num_replicates,
                                                           results_path)
    else:
        log_or_print(
            f'Creating data sets of specified durations: {train_set_durs}',
            logger=logger, level='info'
        )
        # do all subsetting before training, so that we fail early if subsetting is going to fail
        train_dur_csv_paths = train_dur_csv_paths_from_df(dataset_df,
                                                          csv_path,
                                                          train_set_durs,
                                                          timebin_dur,
                                                          num_replicates,
                                                          results_path,
                                                          labelset,
                                                          window_size,
                                                          spect_key,
                                                          timebins_key,
                                                          labelmap,
                                                          logger)

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
