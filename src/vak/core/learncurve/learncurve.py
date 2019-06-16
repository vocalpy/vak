import logging
import os
from pathlib import Path
import sys
from datetime import datetime

import numpy as np

from .train import train
from .test import test

LEARN_CURVE_DIR_STEM = 'learning_curve.'


def learning_curve(train_vds_path,
                   val_vds_path,
                   test_vds_path,
                   total_train_set_duration,
                   train_set_durs,
                   num_replicates,
                   networks,
                   num_epochs,
                   val_error_step=None,
                   checkpoint_step=None,
                   patience=None,
                   save_only_single_checkpoint_file=True,
                   normalize_spectrograms=False,
                   use_train_subsets_from_previous_run=False,
                   previous_run_path=None,
                   save_transformed_data=False,
                   output_dir=None):
    """generate learning curve, by first running learncurve.train
    to train models with a range of training set sizes, and then
    running learncurve.test to measure accuracy of those models on
    a test set

    Parameters
    ----------
    train_vds_path : str
        path to VocalizationDataset that represents training data
    val_vds_path : str
        path to VocalizationDataset that represents validation data
    test_vds_path : str
        path to VocalizationDataset that represents test data
    total_train_set_duration : int
        total duration of training set, in seconds
    train_set_durs : list
        of int, durations in seconds of subsets taken from training data
        to create a learning curve, e.g. [5, 10, 15, 20]
    num_replicates : int
        number of times to replicate training for each training set duration
        to better estimate mean accuracy for a training set of that size.
        Each replicate uses a different randomly drawn subset of the training
        data (but of the same duration).
    networks : dict
        where each key is the name of a neural network and the corresponding
        value is the configuration for that network (in a namedtuple or a dict)
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
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
    normalize_spectrograms : bool
        if True, use utils.spect.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    use_train_subsets_from_previous_run : bool
        if True, use training subsets saved in a previous run
    previous_run_path : str
        path to results directory from a previous run
    save_transformed_data : bool
        if True, save transformed data (i.e. scaled, reshaped). The data can then
        be used on a subsequent run of learncurve (e.g. if you want to compare results
        from different hyperparameters across the exact same training set).
        Also useful if you need to check what the data looks like when fed to networks.
    output_dir : str
        path to directory where results from this run of learncurve should be saved.
        Default is None. If none, then a directory for results will be created in
        the current working directory.

    Returns
    -------
    results_dirname : str
        directory created to hold results
    """
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    if val_error_step and val_vds_path is None:
        raise ValueError(
            f"val_error_step set to {val_error_step} but val_vds_path is None; please provide a path to "
            f"a validation data set that can be used to check error rate very {val_error_step} steps"
        )

    if val_vds_path and val_error_step is None:
        raise ValueError(
            "val_vds_path was provided but val_error_step is None; please provide a value for val_error_step"
        )

    max_train_set_dur = np.max(train_set_durs)
    if max_train_set_dur > total_train_set_duration:
        raise ValueError('Largest duration for a training set of {} '
                         'is greater than total duration of training set, {}'
                         .format(max_train_set_dur, total_train_set_duration))

    if output_dir:
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(
                f'specified output directory not found: {output_dir}'
            )

    # ---------------- logging -----------------------------------------------------------------------------------------
    # need to set up a results dir so we have some place to put the log file
    if output_dir and LEARN_CURVE_DIR_STEM in Path(output_dir).name:
        # (because cli.learncurve made directory already and passed as argument
        # to output dir ... so we don't want to change it)
        results_dirname = output_dir
    else:  # else we need to make a directory
        timenow = datetime.now().strftime('%y%m%d_%H%M%S')
        results_dirname = f'{LEARN_CURVE_DIR_STEM}{timenow}'

        if output_dir:
            results_dirname = os.path.join(output_dir,
                                           results_dirname)
        else:
            results_dirname = os.path.join(os.getcwd(), results_dirname)

    if not os.path.isdir(results_dirname):
        os.makedirs(results_dirname)

    logger = logging.getLogger('learncurve')

    # check whether logger already has config (because this function was called by cli.learncurve)
    # and if it doesn't then configure it
    if logging.getLevelName(logger.level) != 'INFO':
        logger.setLevel('INFO')

    if logging.FileHandler not in [type(handler) for handler in logger.handlers]:
        logfile_name = os.path.join(results_dirname,
                                    'logfile_from_running_learncurve_' + timenow + '.log')
        logger.addHandler(logging.FileHandler(logfile_name))
        logger.info('Logging results to {}'.format(results_dirname))
    if logging.StreamHandler not in [type(handler) for handler in logger.handlers]:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    train(train_vds_path,
          total_train_set_duration,
          train_set_durs,
          num_replicates,
          networks,
          num_epochs,
          results_dirname,
          val_vds_path,
          val_error_step,
          checkpoint_step,
          patience,
          save_only_single_checkpoint_file,
          normalize_spectrograms,
          use_train_subsets_from_previous_run,
          previous_run_path,
          save_transformed_data)

    test(results_dirname,
         test_vds_path,
         train_vds_path,
         networks,
         train_set_durs,
         num_replicates,
         normalize_spectrograms=normalize_spectrograms,
         save_transformed_data=save_transformed_data)

    return results_dirname
