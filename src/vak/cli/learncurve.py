import copy
import logging
import os
import pickle
import shutil
import sys
import warnings
from datetime import datetime
from configparser import ConfigParser

import joblib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .. import network
from .. import utils
from .. import config


def learncurve(train_data_dict_path,
               val_data_dict_path,
               spect_params,
               total_train_set_duration,
               train_set_durs,
               num_replicates,
               networks,
               num_epochs,
               config_file,
               val_error_step=None,
               checkpoint_step=None,
               patience=None,
               save_only_single_checkpoint_file=True,
               normalize_spectrograms=False,
               use_train_subsets_from_previous_run=False,
               previous_run_path=None,
               root_results_dir=None,
               save_transformed_data=False,
               ):
    """train models used by cli.summary to generate learning curve

    Parameters
    ----------
    train_data_dict_path : str
        path to training data
    val_data_dict_path : str
        path to validation data
    spect_params : dict
        parameters for creating spectrograms.
        Used to ensure that what's in config file matches what's in
        the data.
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
    networks : namedtuple
        where each field is the Config tuple for a neural network and the name
        of that field is the name of the class that represents the network.
    num_epochs : int
        number of training epochs. One epoch = one iteration through the entire
        training set.
    config_file : str
        path to config.ini file. Used to rewrite file with options determined by
        this function and needed for other functions (e.g. cli.summary)
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
        if True, use spect.utils.data.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    use_train_subsets_from_previous_run : bool
        if True, use training subsets saved in a previous run
    previous_run_path : str
        path to results directory from a previous run
    root_results_dir : str
        path in which to create results directory for this run of cli.learncurve
    save_transformed_data : bool
        if True, save transformed data (i.e. scaled, reshaped). The data can then
        be used on a subsequent run of learncurve (e.g. if you want to compare results
        from different hyperparameters across the exact same training set).
        Also useful if you need to check what the data looks like when fed to networks.

    Returns
    -------
    None

    Saves results in root_results_dir and adds some options to config_file.
    """
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    if root_results_dir:
        results_dirname = os.path.join(root_results_dir,
                                       'results_' + timenow)
    else:
        results_dirname = os.path.join('.', 'results_' + timenow)
    os.makedirs(results_dirname)
    shutil.copy(config_file, results_dirname)

    logfile_name = os.path.join(results_dirname,
                                'logfile_from_running_learncurve_' + timenow + '.log')
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
    logger.addHandler(logging.FileHandler(logfile_name))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info('Logging results to {}'.format(results_dirname))
    logger.info('Using config file: {}'.format(config_file))

    logger.info('Loading training data from {}'.format(
        os.path.dirname(
            train_data_dict_path)))
    train_data_dict = joblib.load(train_data_dict_path)
    labels_mapping = train_data_dict['labels_mapping']
    if train_data_dict['spect_params'] == 'matlab':
        warnings.warn('Not checking parameters used to compute spectrogram in '
                      'training data,\n because spectrograms were created in '
                      'Matlab')
    else:
        if train_data_dict['spect_params'] != spect_params:
            raise ValueError('Spectrogram parameters in config file '
                             'do not match parameters specified in training data_dict.\n'
                             'Config file is: {}\n'
                             'Data dict is: {}.'.format(config_file,
                                                        train_data_dict_path))

    # save copy of labels_mapping in results directory
    labels_mapping_file = os.path.join(results_dirname, 'labels_mapping')
    with open(labels_mapping_file, 'wb') as labels_map_file_obj:
        pickle.dump(labels_mapping, labels_map_file_obj)

    # n_syllables, i.e., number of label classes to predict
    # Note that mapping includes label for silent gap b/t syllables
    # Error checking code to ensure that it is in fact a consecutive
    # series of integers from 0 to n, so we don't predict classes that
    # don't exist
    if sorted(labels_mapping.values()) != list(range(len(labels_mapping))):
        raise ValueError('Labels mapping does not map to a consecutive'
                         'series of integers from 0 to n (where 0 is the '
                         'silent gap label and n is the number of syllable'
                         'labels).')
    n_syllables = len(labels_mapping)
    logger.debug('n_syllables: '.format(n_syllables))

    # copy training data to results dir so we have it stored with results
    logger.info('copying {} to {}'.format(train_data_dict_path,
                                          results_dirname))
    shutil.copy(train_data_dict_path, results_dirname)

    (X_train,
     Y_train,
     X_train_spect_ID_vector,
     timebin_dur,
     files_used) = (train_data_dict['X_train'],
                    train_data_dict['Y_train'],
                    train_data_dict['spect_ID_vector'],
                    train_data_dict['timebin_dur'],
                    train_data_dict['filenames'])

    logger.info('Size of each timebin in spectrogram, in seconds: {}'
                .format(timebin_dur))
    # dump filenames to a text file
    # to be consistent with what the matlab helper function does
    files_used_filename = os.path.join(results_dirname, 'training_filenames')
    with open(files_used_filename, 'w') as files_used_fileobj:
        files_used_fileobj.write('\n'.join(files_used))

    dur_diff = np.abs((X_train.shape[-1] * timebin_dur) - total_train_set_duration)
    if dur_diff > 1.0:
        raise ValueError('Duration of X_train in seconds from train_data_dict '
                         'is more than one second different from '
                         'duration specified in config file.\n'
                         'train_data_dict: {}\n'
                         'config file: {}'
                         .format(train_data_dict_path, config_file))
    logger.info('Total duration of training set (in s): {}'
                .format(total_train_set_duration))

    max_train_set_dur = np.max(train_set_durs)

    if max_train_set_dur > total_train_set_duration:
        raise ValueError('Largest duration for a training set of {} '
                         'is greater than total duration of training set, {}'
                         .format(max_train_set_dur, total_train_set_duration))

    logger.info('Will train network with training sets of '
                'following durations (in s): {}'.format(train_set_durs))

    # transpose X_train, so rows are timebins and columns are frequency bins
    # because cnn-bilstm network expects this orientation for input
    X_train = X_train.T
    if save_transformed_data:
        joblib.dump(X_train, os.path.join(results_dirname, 'X_train'))
        joblib.dump(Y_train, os.path.join(results_dirname, 'Y_train'))

    REPLICATES = range(num_replicates)
    logger.info('will replicate training {} times for each duration of training set'
                .format(num_replicates))

    val_data_dict = joblib.load(val_data_dict_path)
    (X_val,
     Y_val) = (val_data_dict['X_val'],
               val_data_dict['Y_val'])
    if val_data_dict['spect_params'] == 'matlab':
        warnings.warn('Not checking parameters used to compute spectrogram in '
                      'validation data,\n because spectrograms were created in '
                      'Matlab')
    else:
        if val_data_dict['spect_params'] != spect_params:
            raise ValueError('Spectrogram parameters in config file '
                             'do not match parameters specified in validation data_dict.\n'
                             'Config file is: {}\n'
                             'Data dict is: {}.'.format(config_file,
                                                        val_data_dict_path))
    #####################################################
    # note that we 'transpose' the spectrogram          #
    # so that rows are time and columns are frequencies #
    #####################################################
    X_val = X_val.T
    if save_transformed_data:
        joblib.dump(X_val, os.path.join(results_dirname, 'X_val'))
        joblib.dump(Y_val, os.path.join(results_dirname, 'Y_val'))

    logger.info('will measure error on validation set '
                'every {} steps of training'.format(val_error_step))
    logger.info('will save a checkpoint file '
                'every {} steps of training'.format(checkpoint_step))

    if save_only_single_checkpoint_file:
        logger.info('save_only_single_checkpoint_file = True\n'
                    'will save only one checkpoint file'
                    'and overwrite every {} steps of training'.format(checkpoint_step))
    else:
        logger.info('save_only_single_checkpoint_file = False\n'
                    'will save a separate checkpoint file '
                    'every {} steps of training'.format(checkpoint_step))

    logger.info('\'patience\' is set to: {}'.format(patience))

    logger.info('number of training epochs will be {}'
                .format(num_epochs))

    if normalize_spectrograms:
        logger.info('will normalize spectrograms for each training set')
        # need a copy of X_val when we normalize it below
        X_val_copy = copy.deepcopy(X_val)

    NETWORKS = network._load()

    for train_set_dur in train_set_durs:
        for replicate in REPLICATES:
            logger.info("training with training set duration of {} seconds,"
                        "replicate #{}".format(train_set_dur, replicate))
            training_records_dir = ('records_for_training_set_with_duration_of_'
                                    + str(train_set_dur) + '_sec_replicate_'
                                    + str(replicate))
            training_records_path = os.path.join(results_dirname,
                                                 training_records_dir)

            if not os.path.isdir(training_records_path):
                os.makedirs(training_records_path)

            if use_train_subsets_from_previous_run:
                train_inds_path = os.path.join(previous_run_path,
                                               training_records_dir,
                                               'train_inds')
                with open(train_inds_path, 'rb') as f:
                    train_inds = pickle.load(f)
            else:
                train_inds = utils.data.get_inds_for_dur(X_train_spect_ID_vector,
                                                         Y_train,
                                                         labels_mapping,
                                                         train_set_dur,
                                                         timebin_dur)
            with open(os.path.join(training_records_path, 'train_inds'),
                      'wb') as train_inds_file:
                pickle.dump(train_inds, train_inds_file)
            X_train_subset = X_train[train_inds, :]
            Y_train_subset = Y_train[train_inds]
            if Y_train_subset.ndim > 1:
                # not clear to me right why labeled_timebins get saved as (n, 1)
                # instead of as (n) vector--i.e. if another functions depends on that shape
                # Below is hackish way around figuring that out.
                Y_train_subset = np.squeeze(Y_train_subset)

            if normalize_spectrograms:
                spect_scaler = utils.data.SpectScaler()
                X_train_subset = spect_scaler.fit_transform(X_train_subset)
                logger.info('normalizing validation set to match training set')
                X_val = spect_scaler.transform(X_val_copy)
                scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                               .format(train_set_dur, replicate))
                joblib.dump(spect_scaler,
                            os.path.join(training_records_path, scaler_name))

            scaled_data_filename = os.path.join(training_records_path,
                                                'scaled_spects_duration_{}_replicate_{}'
                                                .format(train_set_dur, replicate))
            scaled_data_dict = {'X_train_subset_scaled': X_train_subset,
                                'X_val_scaled': X_val,
                                'Y_train_subset': Y_train_subset}
            joblib.dump(scaled_data_dict, scaled_data_filename)

            freq_bins = X_train_subset.shape[-1]  # number of columns
            logger.debug('freq_bins in spectrogram: '.format(freq_bins))

            for net_name, net_config in zip(networks._fields, networks):
                net_config_dict = net_config._asdict()
                net_config_dict['n_syllables'] = n_syllables
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

                (X_val_batch,
                 Y_val_batch,
                 num_batches_val) = utils.data.reshape_data_for_batching(X_val,
                                                                         Y_val,
                                                                         net_config.batch_size,
                                                                         net_config.time_bins)

                # save scaled reshaped data
                scaled_reshaped_data_filename = os.path.join(training_records_path,
                                                             'scaled_reshaped_spects_duration_{}_replicate_{}'
                                                             .format(train_set_dur, replicate))
                scaled_reshaped_data_dict = {'X_train_subset_scaled_reshaped': X_train_subset,
                                             'Y_train_subset_reshaped': Y_train_subset,
                                             'X_val_scaled_batch': X_val_batch,
                                             'Y_val_batch': Y_val_batch}
                joblib.dump(scaled_reshaped_data_dict, scaled_reshaped_data_filename)

                costs = []
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

                    for epoch in range(num_epochs):
                        # every epoch we are going to shuffle the order in which we look at every window
                        shuffle_order = np.random.permutation(num_windows)
                        shuffle_order = shuffle_order[:new_last_ind].reshape(num_batches, net_config.batch_size)
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
                            _cost, _, summary = sess.run((net.cost,
                                                          net.optimize,
                                                          net.merged_summary_op),
                                                feed_dict=d)
                            costs.append(_cost)
                            net.summary_writer.add_summary(summary, epoch)
                            pbar.set_description(
                                f"epoch {epoch + 1}, batch {batch_num + 1} of {num_batches}, cost: {_cost:8.4f}"
                            )

                        if val_error_step:
                            if epoch % val_error_step == 0:
                                if 'Y_pred_val' in locals():
                                    del Y_pred_val

                                for b in range(num_batches_val):  # "b" is "batch number"
                                    X_b = X_val_batch[:, b * net_config.time_bins: (b + 1) * net_config.time_bins, :]
                                    Y_b = Y_val_batch[:, b * net_config.time_bins: (b + 1) * net_config.time_bins]
                                    d = {net.X: X_b,
                                         net.y: Y_b,
                                         net.lng: [net_config.time_bins] * net_config.batch_size}

                                    if 'Y_pred_val' in locals():
                                        preds = sess.run(net.predict, feed_dict=d)
                                        preds = preds.reshape(net_config.batch_size, -1)
                                        Y_pred_val = np.concatenate((Y_pred_val, preds), axis=1)
                                    else:
                                        Y_pred_val = sess.run(net.predict, feed_dict=d)
                                        Y_pred_val = Y_pred_val.reshape(net_config.batch_size, -1)

                                # get rid of zero padding predictions
                                Y_pred_val = Y_pred_val.ravel()[:Y_val.shape[0], np.newaxis]
                                val_errs.append(np.sum(Y_pred_val - Y_val != 0) / Y_val.shape[0])
                                print("epoch {}, validation error: {}".format(epoch + 1, val_errs[-1]))

                            if patience:
                                if val_errs[-1] < curr_min_err:
                                    # error went down, set as new min and reset counter
                                    curr_min_err = val_errs[-1]
                                    err_patience_counter = 0
                                    print("Validation error improved.\n"
                                          "Saving checkpoint to {}".format(checkpoint_path))
                                    net.saver.save(sess, os.path.join(checkpoint_path, checkpoint_filename))
                                else:
                                    err_patience_counter += 1
                                    if err_patience_counter > patience:
                                        print("stopping because validation error has not improved in {} epochs"
                                              .format(patience))
                                        with open(os.path.join(training_records_path, "costs"), 'wb') as costs_file:
                                            pickle.dump(costs, costs_file)
                                        with open(os.path.join(training_records_path, "val_errs"), 'wb') as val_errs_file:
                                            pickle.dump(val_errs, val_errs_file)
                                        break

                        if save_only_single_checkpoint_file is False:
                            checkpoint_path_tmp = os.path.join(checkpoint_path,
                                                               checkpoint_filename + '_epoch_{}'.format(epoch))
                        else:
                            checkpoint_path_tmp = os.path.join(checkpoint_path, checkpoint_filename)

                        if checkpoint_step:
                            if epoch % checkpoint_step == 0:
                                "Saving checkpoint."
                                net.saver.save(sess, checkpoint_path_tmp)
                                with open(os.path.join(training_records_path, "val_errs"), 'wb') as val_errs_file:
                                    pickle.dump(val_errs, val_errs_file)

                        if epoch == (num_epochs-1):  # if this is the last epoch
                            "Reached max. number of epochs, saving checkpoint."
                            net.saver.save(sess, checkpoint_path_tmp)
                            with open(os.path.join(results_dirname_this_net, "costs"),
                                      'wb') as costs_file:
                                pickle.dump(costs, costs_file)
                            with open(os.path.join(results_dirname_this_net, "val_errs"),
                                      'wb') as val_errs_file:
                                pickle.dump(val_errs, val_errs_file)


    # lastly rewrite config file,
    # so that paths where results were saved are automatically in config
    config = ConfigParser()
    config.read(config_file)
    config.set(section='DATA',
               option='n_syllables',
               value=str(n_syllables))
    config.set(section='OUTPUT',
               option='results_dir_made_by_main_script',
               value=results_dirname)
    with open(config_file, 'w') as config_file_rewrite:
        config.write(config_file_rewrite)


if __name__ == "__main__":
    config_file = os.path.normpath(sys.argv[1])
    config = config.parse.parse_config(config_file)
    learncurve(train_data_dict_path=config.train.train_data_dict_path,
               val_data_dict_path=config.train.val_data_dict_path,
               spect_params=config.spect_params,
               total_train_set_duration=config.data.total_train_set_dur,
               train_set_durs=config.train.train_set_durs,
               num_replicates=config.train.num_replicates,
               num_epochs=config.train.num_epochs,
               config_file=config_file,
               networks=config.networks,
               val_error_step=config.train.val_error_step,
               checkpoint_step=config.train.checkpoint_step,
               patience=config.train.patience,
               save_only_single_checkpoint_file=config.train.save_only_single_checkpoint_file,
               normalize_spectrograms=config.train.normalize_spectrograms,
               use_train_subsets_from_previous_run=config.train.use_train_subsets_from_previous_run,
               previous_run_path=config.train.previous_run_path,
               root_results_dir=config.output.root_results_dir)
