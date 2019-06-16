import copy
import logging
import os
import pickle
from math import isclose

import joblib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ...utils.general import safe_truncate
from ...utils.spect import SpectScaler
from ... import network
from ... import utils
from ...dataset.classes import VocalizationDataset


def train(train_vds_path,
          total_train_set_duration,
          train_set_durs,
          num_replicates,
          networks,
          num_epochs,
          output_dir,
          val_vds_path=None,
          val_error_step=None,
          checkpoint_step=None,
          patience=None,
          save_only_single_checkpoint_file=True,
          normalize_spectrograms=False,
          use_train_subsets_from_previous_run=False,
          previous_run_path=None,
          save_transformed_data=False,
          measure_train_err=True):
    """train models used by vak.core.learncurve.test to generate learning curve

    Parameters
    ----------
    train_vds_path : str
        path to VocalizationDataset that represents training data
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
    output_dir : str
        name of directory where results from this run of 'train' will be saved.
    val_vds_path : str
        path to VocalizationDataset that represents validation data
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

    Returns
    -------
    None
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

    train_dirname = os.path.join(output_dir, 'train')
    os.makedirs(train_dirname)

    # ---------------- logging -----------------------------------------------------------------------------------------
    logger = logging.getLogger('learncurve.train')

    if logging.getLevelName(logger.level) != 'INFO':
        logger.setLevel('INFO')

    # ---------------- load training data  -----------------------------------------------------------------------------
    logger.info('Loading training VocalizationDataset from {}'.format(
        os.path.dirname(
            train_vds_path)))
    train_vds = VocalizationDataset.load(json_fname=train_vds_path)

    if train_vds.are_spects_loaded() is False:
        train_vds = train_vds.load_spects()

    X_train = train_vds.spects_list()
    X_train_spect_ID_vector = np.concatenate(
        [np.ones((spect.shape[-1],), dtype=np.int64) * ind for ind, spect in enumerate(X_train)]
    )
    X_train = np.concatenate(X_train, axis=1)

    Y_train = train_vds.lbl_tb_list()
    Y_train = np.concatenate(Y_train)

    n_classes = len(train_vds.labelmap)
    logger.debug('n_classes: '.format(n_classes))

    timebin_dur = set([voc.metaspect.timebin_dur for voc in train_vds.voc_list])
    if len(timebin_dur) > 1:
        raise ValueError(
            f'found more than one time bin duration in training VocalizationDataset: {timebin_dur}'
        )
    elif len(timebin_dur) == 1:
        timebin_dur = timebin_dur.pop()
        logger.info('Size of each timebin in spectrogram, in seconds: {timebin_dur}')
    else:
        raise ValueError(
            f'invalid time bin durations from training set: {timebin_dur}'
        )

    X_train_dur = X_train.shape[-1] * timebin_dur
    if not isclose(X_train_dur, total_train_set_duration):
        if X_train_dur > total_train_set_duration:
            try:
                X_train, Y_train, X_train_spect_ID_vector = safe_truncate(X_train,
                                                                          Y_train,
                                                                          X_train_spect_ID_vector,
                                                                          train_vds.labelmap,
                                                                          total_train_set_duration,
                                                                          timebin_dur)
            except ValueError:
                raise ValueError(f'Duration of X_train in seconds, {X_train_dur},from train_vds was greater than '
                                 f'duration specified in config file, {total_train_set_duration} seconds.\n'
                                 f'Was not able to truncate in a way that maintains all classes in dataset.')
        else:
            raise ValueError(f'Duration of X_train in seconds, {X_train_dur},from train_vds is less than '
                             f'duration specified in config file, {total_train_set_duration} seconds.\n')

        X_train_dur = X_train.shape[-1] * timebin_dur

    logger.info(
        f'Total duration of training set (in s): {X_train_dur}'
    )

    # transpose X_train, so rows are timebins and columns are frequency bins
    # because networks expect this orientation for input
    X_train = X_train.T
    if save_transformed_data:
        joblib.dump(X_train, os.path.join(train_dirname, 'X_train'))
        joblib.dump(Y_train, os.path.join(train_dirname, 'Y_train'))

    logger.info(
        f'Will train network with training sets of following durations (in s): {train_set_durs}'
    )

    logger.info(
        f'will replicate training {num_replicates} times for each duration of training set'
    )

    # ---------------- grab all indices for subsets of training data *before* doing any training -------------------
    # we want to fail here, rather than later in the middle of training networks
    logger.info("getting all randomly-drawn subsets of training data before starting training")
    with tqdm(total=len(train_set_durs) * num_replicates) as pbar:
        train_inds_dict = {}

        for train_set_dur in train_set_durs:
            train_inds_dict[train_set_dur] = {}
            for replicate in range(1, num_replicates + 1):
                train_records_dirname = ('records_for_training_set_with_duration_of_'
                                         + str(train_set_dur) + '_sec_replicate_'
                                         + str(replicate))

                pbar.set_description(
                    f"Getting indices for training subset with duration {train_set_dur}, replicate {replicate}"
                )

                if use_train_subsets_from_previous_run:
                    train_inds_path = os.path.join(previous_run_path,
                                                   train_records_dirname,
                                                   'train_inds')
                    logger.info(
                        f"loading indices for training subset with duration {train_set_dur}, replicate {replicate} "
                        f"from: {train_inds_path}"
                    )
                    with open(train_inds_path, 'rb') as f:
                        train_inds = pickle.load(f)

                else:  # if not re-using subsets, need to generate them
                    logger.info(
                        f"generating indices for training subset with duration {train_set_dur}, replicate {replicate}"
                    )

                    training_records_path = os.path.join(train_dirname,
                                                         train_records_dirname)

                    if not os.path.isdir(training_records_path):
                        os.makedirs(training_records_path)

                    train_inds = utils.data.get_inds_for_dur(X_train_spect_ID_vector,
                                                             Y_train,
                                                             train_vds.labelmap,
                                                             train_set_dur,
                                                             timebin_dur)
                    with open(os.path.join(training_records_path, 'train_inds'),
                              'wb') as train_inds_file:
                        pickle.dump(train_inds, train_inds_file)

                train_inds_dict[train_set_dur][replicate] = train_inds
                pbar.update(1)

    # ---------------- load validation set (if there is one) -----------------------------------------------------------
    if val_vds_path:
        val_vds = VocalizationDataset.load(json_fname=val_vds_path)

        if val_vds.are_spects_loaded() is False:
            val_vds = val_vds.load_spects()

        if not val_vds.labelset == train_vds.labelset:
            raise ValueError(
                f'set of labels for validation set, {val_vds.labelset}, '
                f'does not match set for training set: {train_vds.labelset}'
            )

        X_val = val_vds.spects_list()
        X_val = np.concatenate(X_val, axis=1)

        Y_val = val_vds.lbl_tb_list()
        Y_val = np.concatenate(Y_val)

        timebin_dur_val = set([voc.metaspect.timebin_dur for voc in val_vds.voc_list])
        if len(timebin_dur_val) > 1:
            raise ValueError(
                f'found more than one time bin duration in validation VocalizationDataset: {timebin_dur_val}'
            )
        elif len(timebin_dur_val) == 1:
            timebin_dur_val = timebin_dur_val.pop()
            if timebin_dur_val != timebin_dur:
                raise ValueError(
                    f'time bin duration in validation VocalizationDataset, {timebin_dur_val}, did not match that of '
                    f'training set: {timebin_dur}'
                )

        X_val_dur = X_val.shape[-1] * timebin_dur
        logger.info(
            f'Total duration of validation set (in s): {X_val_dur}'
        )

        #####################################################
        # note that we 'transpose' the spectrogram          #
        # so that rows are time and columns are frequencies #
        #####################################################
        X_val = X_val.T
        if save_transformed_data:
            joblib.dump(X_val, os.path.join(train_dirname, 'X_val'))
            joblib.dump(Y_val, os.path.join(train_dirname, 'Y_val'))

        logger.info(
            f'will measure error on validation set every {val_error_step} steps of training'
        )
    else:
        X_val = None  # so we can just check 'if X_val' below, clearer than e.g. 'if val_vds_path'

    logger.info(
        f'will save a checkpoint file every {checkpoint_step} steps of training'
    )

    if save_only_single_checkpoint_file:
        logger.info('save_only_single_checkpoint_file = True\n'
                    'will save only one checkpoint file'
                    f'and overwrite every {checkpoint_step} steps of training')
    else:
        logger.info('save_only_single_checkpoint_file = False\n'
                    'will save a separate checkpoint file '
                    f'every {checkpoint_step} steps of training')

    logger.info(f'\'patience\' is set to: {patience}')

    logger.info(f'number of training epochs will be {num_epochs}')

    if normalize_spectrograms:
        logger.info('will normalize spectrograms for each training set')
        if X_val is not None:
            # need a copy of X_val when we normalize it below
            X_val_copy = copy.deepcopy(X_val)

    NETWORKS = network._load()

    for train_set_dur in train_set_durs:
        for replicate in range(1, num_replicates + 1):
            logger.info("training with training set duration of {} seconds,"
                        "replicate #{}".format(train_set_dur, replicate))
            train_records_dirname = ('records_for_training_set_with_duration_of_'
                                     + str(train_set_dur) + '_sec_replicate_'
                                     + str(replicate))
            training_records_path = os.path.join(train_dirname,
                                                 train_records_dirname)

            train_inds = train_inds_dict[train_set_dur][replicate]
            X_train_subset = X_train[train_inds, :]
            Y_train_subset = Y_train[train_inds]
            if Y_train_subset.ndim > 1:
                # not clear to me right why labeled_timebins get saved as (n, 1)
                # instead of as (n) vector--i.e. if another functions depends on that shape
                # Below is hackish way around figuring that out.
                Y_train_subset = np.squeeze(Y_train_subset)

            if normalize_spectrograms:
                spect_scaler = SpectScaler()
                X_train_subset = spect_scaler.fit_transform(X_train_subset)
                if X_val is not None:
                    logger.info('normalizing validation set to match training set')
                    X_val = spect_scaler.transform(X_val_copy)
                scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                               .format(train_set_dur, replicate))
                joblib.dump(spect_scaler,
                            os.path.join(training_records_path, scaler_name))

            if save_transformed_data:
                scaled_data_filename = os.path.join(training_records_path,
                                                    'scaled_spects_duration_{}_replicate_{}'
                                                    .format(train_set_dur, replicate))
                scaled_data_dict = {'X_train_subset_scaled': X_train_subset,
                                    'Y_train_subset': Y_train_subset}
                if X_val is not None:
                    scaled_data_dict['X_val_scaled'] = X_val
                joblib.dump(scaled_data_dict, scaled_data_filename)

            freq_bins = X_train_subset.shape[-1]  # number of columns
            logger.debug('freq_bins in spectrogram: '.format(freq_bins))

            for net_name, net_config in networks.items():
                net_config_dict = net_config._asdict()
                net_config_dict['n_syllables'] = n_classes
                if 'freq_bins' in net_config_dict:
                    net_config_dict['freq_bins'] = freq_bins
                net = NETWORKS[net_name](**net_config_dict)

                results_dirname_this_net = os.path.join(training_records_path, net_name)

                if not os.path.isdir(results_dirname_this_net):
                    os.makedirs(results_dirname_this_net)
                logs_subdir = ('log_{}_train_set_with_duration_of_{}_sec_replicate_{}'
                               .format(net_name, str(train_set_dur), str(replicate)))
                logs_path = os.path.join(results_dirname_this_net,
                                         'logs',
                                         logs_subdir)
                if not os.path.isdir(logs_path):
                    os.makedirs(logs_path)

                checkpoint_path = os.path.join(results_dirname_this_net, 'checkpoints')
                if not os.path.isdir(checkpoint_path):
                    os.makedirs(checkpoint_path)

                checkpoint_filename = ('checkpoint_{}_train_set_dur_{}_sec_replicate_{}'
                                       .format(net_name, str(train_set_dur), str(replicate)))

                net.add_summary_writer(logs_path=logs_path)

                if X_val is not None:
                    (X_val_batch,
                     Y_val_batch,
                     num_batches_val) = utils.data.reshape_data_for_batching(X_val,
                                                                             net_config.batch_size,
                                                                             net_config.time_bins,
                                                                             Y_val)

                if save_transformed_data:
                    scaled_reshaped_data_filename = os.path.join(training_records_path,
                                                                 'scaled_reshaped_spects_duration_{}_replicate_{}'
                                                                 .format(train_set_dur, replicate))
                    scaled_reshaped_data_dict = {'X_train_subset_scaled_reshaped': X_train_subset,
                                                 'Y_train_subset_reshaped': Y_train_subset}
                    if X_val is not None:
                        scaled_reshaped_data_dict['X_val_scaled_batch'] = X_val_batch
                        scaled_reshaped_data_dict['Y_val_batch'] = Y_val_batch

                        joblib.dump(scaled_reshaped_data_dict, scaled_reshaped_data_filename)

                loss_history = []
                val_errs = []
                curr_min_err = 1  # i.e. 100%
                err_patience_counter = 0

                with tf.Session(graph=net.graph) as sess:
                    sess.run(net.init)

                    # figure out number of batches we can get out of subset of training data
                    # if we slide a window along the spectrogram with a stride of 1
                    # and use each window as one sample in a batch
                    num_timebins_training_set = X_train_subset.shape[0]
                    num_windows = num_timebins_training_set - net_config.time_bins
                    logger.info('training set with {} time bins will yield {} windows '
                                'of width {} time bins'
                                .format(num_timebins_training_set,
                                        num_windows,
                                        net_config.time_bins))
                    num_batches = num_windows // net_config.batch_size  # note floor division
                    # meaning we'll throw away some windows
                    new_last_ind = net_config.batch_size * num_batches
                    logger.info('divided into batches of size {} yields {} batches '
                                'for a total of {} windows (throwing away {}).'
                                .format(net_config.batch_size,
                                        num_batches,
                                        new_last_ind,
                                        num_windows - new_last_ind))

                    for epoch in range(1, num_epochs + 1):
                        # every epoch we are going to shuffle the order in which we look at every window
                        shuffle_order = np.random.permutation(num_windows)
                        shuffle_order = shuffle_order[:new_last_ind].reshape(num_batches, net_config.batch_size)
                        total_loss = 0  # per epoch
                        pbar = tqdm(shuffle_order)
                        for batch_num, batch_inds in enumerate(pbar):
                            X_batch = []
                            Y_batch = []
                            for start_ind in batch_inds:
                                X_batch.append(
                                    X_train_subset[start_ind:start_ind+net_config.time_bins, :]
                                )
                                Y_batch.append(
                                    Y_train_subset[start_ind:start_ind+net_config.time_bins]
                                )
                            X_batch = np.stack(X_batch)
                            Y_batch = np.stack(Y_batch)
                            d = {net.X: X_batch,
                                 net.y: Y_batch,
                                 net.lng: [net_config.time_bins] * net_config.batch_size}
                            loss, _, summary = sess.run((net.cost, net.optimize, net.merged_summary_op), feed_dict=d)
                            total_loss += loss
                            net.summary_writer.add_summary(summary, epoch)
                            pbar.set_description(
                                f"epoch {epoch}, batch {batch_num + 1} of {num_batches}, loss: {loss:7.3f}"
                            )

                        avg_loss = total_loss / num_batches
                        logger.info(f"Epoch {epoch}, average loss: {avg_loss:7.3f}")
                        loss_history.append(avg_loss)

                        if measure_train_err is True:
                            (X_tr_batch,
                             Y_tr_batch,
                             num_batches_tr) = utils.data.reshape_data_for_batching(X_train_subset,
                                                                                    net_config.batch_size,
                                                                                    net_config.time_bins,
                                                                                    Y_train_subset[:, np.newaxis])
                            Y_pred_tr = []
                            for b in range(num_batches_tr):  # "b" is "batch number"
                                X_b = X_tr_batch[:, b * net_config.time_bins: (b + 1) * net_config.time_bins, :]
                                d = {net.X: X_b,
                                     net.lng: [net_config.time_bins] * net_config.batch_size}

                                preds = sess.run(net.predict, feed_dict=d)
                                preds = preds.reshape(net_config.batch_size, -1)
                                Y_pred_tr.append(preds)

                            Y_pred_tr = np.concatenate(Y_pred_tr, axis=1)
                            # get rid of zero padding predictions
                            Y_pred_tr = Y_pred_tr.ravel()[:Y_train_subset.shape[0]]
                            train_err = np.sum(Y_pred_tr != Y_train_subset) / Y_train_subset.shape[0]
                            logger.info(
                                f"epoch {epoch}, training error: {train_err:7.3f}"
                            )

                        if X_val is not None:
                            if epoch % val_error_step == 0:
                                Y_pred_val = []

                                for b in range(num_batches_val):  # "b" is "batch number"
                                    X_b = X_val_batch[:, b * net_config.time_bins: (b + 1) * net_config.time_bins, :]
                                    d = {net.X: X_b,
                                         net.lng: [net_config.time_bins] * net_config.batch_size}

                                    preds = sess.run(net.predict, feed_dict=d)
                                    preds = preds.reshape(net_config.batch_size, -1)
                                    Y_pred_val.append(preds)

                                Y_pred_val = np.concatenate(Y_pred_val, axis=1)
                                # get rid of zero padding predictions
                                Y_pred_val = Y_pred_val.ravel()[:Y_val.shape[0], np.newaxis]
                                val_errs.append(np.sum(Y_pred_val != Y_val) / Y_val.shape[0])
                                logger.info(
                                    f"epoch {epoch}, validation error: {val_errs[-1]:7.3f}"
                                )

                                if patience:
                                    if val_errs[-1] < curr_min_err:
                                        # error went down, set as new min and reset counter
                                        curr_min_err = val_errs[-1]
                                        err_patience_counter = 0
                                        logger.info("Validation error improved.\n"
                                                    "Saving checkpoint to {}".format(checkpoint_path))
                                        net.saver.save(sess, os.path.join(checkpoint_path, checkpoint_filename))
                                    else:
                                        err_patience_counter += 1
                                        if err_patience_counter > patience:
                                            logger.info(
                                                "stopping because validation error has not improved "
                                                f"after checking validation error {patience} times"
                                                  )
                                            with open(os.path.join(training_records_path, "loss"), 'wb') as loss_file:
                                                pickle.dump(loss_history, loss_file)
                                            with open(
                                                    os.path.join(
                                                        training_records_path, "val_errs"), 'wb') as val_errs_file:
                                                pickle.dump(val_errs, val_errs_file)
                                            break

                        if save_only_single_checkpoint_file is False:
                            checkpoint_path_tmp = os.path.join(checkpoint_path,
                                                               checkpoint_filename + '_epoch_{}'.format(epoch))
                        else:
                            checkpoint_path_tmp = os.path.join(checkpoint_path, checkpoint_filename)

                        if checkpoint_step:
                            if epoch % checkpoint_step == 0:
                                logger.info("Saving checkpoint.")
                                net.saver.save(sess, checkpoint_path_tmp)
                                with open(os.path.join(training_records_path, "val_errs"), 'wb') as val_errs_file:
                                    pickle.dump(val_errs, val_errs_file)

                        if epoch == (num_epochs-1):  # if this is the last epoch
                            logger.info("Reached max. number of epochs, saving checkpoint.")
                            net.saver.save(sess, checkpoint_path_tmp)
                            with open(os.path.join(results_dirname_this_net, "loss"),
                                      'wb') as loss_file:
                                pickle.dump(loss_history, loss_file)
                            with open(os.path.join(results_dirname_this_net, "val_errs"),
                                      'wb') as val_errs_file:
                                pickle.dump(val_errs, val_errs_file)

