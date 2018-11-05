import copy
import logging
import os
import pickle
import shutil
import sys
import warnings
from configparser import ConfigParser, NoOptionError
from datetime import datetime

import joblib
import numpy as np
import tensorflow as tf

import tweetynet
from tweetynet import TweetyNet


def train(config_file):
    """train a single models using training set specified in config.ini file"""
    if not config_file.endswith('.ini'):
        raise ValueError('{} is not a valid config file, '
                         'must have .ini extension'.format(config_file))
    if not os.path.isfile(config_file):
        raise FileNotFoundError('config file {} is not found'
                                .format(config_file))
    config = ConfigParser()
    config.read(config_file)

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    if config.has_option('OUTPUT', 'root_results_dir'):
        root_results_dir = config['OUTPUT']['root_results_dir']
        results_dirname = os.path.join(root_results_dir,
                                       'results_' + timenow)
    else:
        results_dirname = os.path.join('.', 'results_' + timenow)
    os.makedirs(results_dirname)
    # copy config file into results dir now that we've made the dir
    shutil.copy(config_file, results_dirname)

    logfile_name = os.path.join(results_dirname,
                                'logfile_from_running_main_' + timenow + '.log')
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
    logger.addHandler(logging.FileHandler(logfile_name))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info('Logging results to {}'.format(results_dirname))
    logger.info('Using config file: {}'.format(config_file))

    train_data_dict_path = config['TRAIN']['train_data_path']
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
        # require user to specify parameters for spectrogram
        # instead of having defaults (as was here previously)
        # helps ensure we don't mix up different params
        spect_params = {}
        spect_params['fft_size'] = int(config['SPECTROGRAM']['fft_size'])
        spect_params['step_size'] = int(config['SPECTROGRAM']['step_size'])
        spect_params['freq_cutoffs'] = [float(element)
                                        for element in
                                        config['SPECTROGRAM']['freq_cutoffs']
                                            .split(',')]
        if config.has_option('SPECTROGRAM', 'thresh'):
            spect_params['thresh'] = float(config['SPECTROGRAM']['thresh'])
        if config.has_option('SPECTROGRAM', 'transform_type'):
            spect_params['transform_type'] = config['SPECTROGRAM']['transform_type']

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

    total_train_set_duration = float(config['DATA']['total_train_set_duration'])
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

    logger.info('Will train network with training sets of '
                'following durations (in s): {}'.format(TRAIN_SET_DURS))

    # transpose X_train, so rows are timebins and columns are frequency bins
    # because cnn-bilstm network expects this orientation for input
    X_train = X_train.T

    val_data_dict_path = config['TRAIN']['val_data_path']
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
    joblib.dump(X_val, os.path.join(results_dirname, 'X_val'))
    joblib.dump(Y_val, os.path.join(results_dirname, 'Y_val'))

    val_error_step = int(config['TRAIN']['val_error_step'])
    logger.info('will measure error on validation set '
                'every {} steps of training'.format(val_error_step))
    checkpoint_step = int(config['TRAIN']['checkpoint_step'])
    logger.info('will save a checkpoint file '
                'every {} steps of training'.format(checkpoint_step))
    save_only_single_checkpoint_file = config.getboolean('TRAIN',
                                                         'save_only_single_checkpoint_file')
    if save_only_single_checkpoint_file:
        logger.info('save_only_single_checkpoint_file = True\n'
                    'will save only one checkpoint file'
                    'and overwrite every {} steps of training'.format(checkpoint_step))
    else:
        logger.info('save_only_single_checkpoint_file = False\n'
                    'will save a separate checkpoint file '
                    'every {} steps of training'.format(checkpoint_step))

    patience = config['TRAIN']['patience']
    try:
        patience = int(patience)
    except ValueError:
        if patience == 'None':
            patience = None
        else:
            raise TypeError('patience must be an int or None, but'
                            'is {} and parsed as type {}'
                            .format(patience, type(patience)))
    logger.info('\'patience\' is set to: {}'.format(patience))

    # set params used for sending data to graph in batches
    batch_size = int(config['NETWORK']['batch_size'])
    time_steps = int(config['NETWORK']['time_steps'])
    logger.info('will train network with batches of size {}, '
                'where each spectrogram in batch contains {} time steps'
                .format(batch_size, time_steps))

    n_max_iter = int(config['TRAIN']['n_max_iter'])
    logger.info('maximum number of training steps will be {}'
                .format(n_max_iter))

    normalize_spectrograms = config.getboolean('TRAIN', 'normalize_spectrograms')
    if normalize_spectrograms:
        logger.info('will normalize spectrograms for each training set')
        # need a copy of X_val when we normalize it below
        X_val_copy = copy.deepcopy(X_val)

    ### start of actual training ###
    costs = []
    val_errs = []
    curr_min_err = 1  # i.e. 100%
    err_patience_counter = 0

    logger.info("training model.")
    training_records_dir = 'records_for_training'
    training_records_path = os.path.join(results_dirname,
                                         training_records_dir)

    if not os.path.isdir(training_records_path):
        os.makedirs(training_records_path)
    checkpoint_filename = 'checkpoint_'

    if normalize_spectrograms:
        spect_scaler = tweetynet.utils.SpectScaler()
        X_train = spect_scaler.fit_transform(X_train)
        logger.info('normalizing validation set to match training set')
        X_val = spect_scaler.transform(X_val_copy)
        scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                       .format(train_set_dur, replicate))
        joblib.dump(spect_scaler,
                    os.path.join(results_dirname, scaler_name))

    scaled_data_filename = os.path.join(training_records_path,
                                        'scaled_spects_duration_{}_replicate_{}'
                                        .format(train_set_dur, replicate))
    scaled_data_dict = {'X_train_subset_scaled': X_train,
                        'X_val_scaled': X_val,
                        'Y_train_subset': Y_train}
    joblib.dump(scaled_data_dict, scaled_data_filename)

    # reshape data for network
    batch_spec_rows = len(train_inds) // batch_size

    # this is the original way reshaping was done
    # note that reshaping this way can truncate data set
    X_train = \
        X_train[0:batch_spec_rows * batch_size].reshape((batch_size,
                                                         batch_spec_rows,
                                                         -1))
    Y_train = \
        Y_train[0:batch_spec_rows * batch_size].reshape((batch_size, -1))
    reshape_size = Y_train.ravel().shape[-1]
    diff = Y_train.shape[-1] - reshape_size
    logger.info('Number of time bins after '
                'reshaping training data: {}.'.format(reshape_size))
    logger.info('Number of time bins less '
                'than specified {}: {}'.format(Y_train.shape[-1],
                                               diff))
    logger.info('Difference in seconds: {}'.format(diff * timebin_dur))

    # note that X_train_subset has shape of (batch, time_bins, frequency_bins)
    # so we permute starting indices from the number of time_bins
    # i.e. X_train_subset.shape[1]
    iter_order = np.random.permutation(X_train.shape[1] - time_steps)
    if len(iter_order) > n_max_iter:
        iter_order = iter_order[0:n_max_iter]
    with open(
            os.path.join(training_records_path,
                         "iter_order"),
            'wb') as iter_order_file:
        pickle.dump(iter_order, iter_order_file)

    input_vec_size = X_train_subset.shape[-1]  # number of columns
    logger.debug('input vec size: '.format(input_vec_size))

    (X_val_batch,
     Y_val_batch,
     num_batches_val) = tweetynet.utils.reshape_data_for_batching(X_val,
                                                                  Y_val,
                                                                  batch_size,
                                                                  time_steps,
                                                                  input_vec_size)

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
    learning_rate = float(config['NETWORK']['learning_rate'])
    logger.debug('learning rate: '.format(learning_rate))

    # rewrite config file
    # to include parameters determined programatically
    config.set(section='NETWORK',
               option='input_vec_size',
               value=str(input_vec_size))
    config.set(section='NETWORK',
               option='n_syllables',
               value=str(n_syllables))
    config.set(section='OUTPUT',
               option='results_dir_made_by_main_script',
               value=results_dirname)
    with open(config_file, 'w') as config_file_rewrite:
        config.write(config_file_rewrite)

    logger.debug('creating graph')

    model = TweetyNet(n_syllables=n_syllables,
                      batch_size=batch_size,
                      input_vec_size=input_vec_size)

    logs_subdir = ('log_training_set_with_duration_of_'
                   + str(train_set_dur) + '_sec_replicate_'
                   + str(replicate))
    logs_path = os.path.join(results_dirname,
                             'logs',
                             logs_subdir)
    if not os.path.isdir(logs_path):
        os.makedirs(logs_path)

    model.add_summary_writer(logs_path=logs_path)

    with tf.Session(graph=model.graph,
                    config=tf.ConfigProto(
                        log_device_placement=True
                        # intra_op_parallelism_threads=512
                    )) as sess:

        # Run the Op to initialize the variables.
        sess.run(model.init)

        # Start the training loop.

        step = 1
        iter_counter = 0

        # loop through training data forever
        # or until validation accuracy stops improving
        # whichever comes first
        while True:
            iternum = iter_order[iter_counter]
            iter_counter = iter_counter + 1
            if iter_counter == len(iter_order):
                iter_counter = 0
            d = {model.X: X_train[:, iternum:iternum + time_steps, :],
                 model.y: Y_train[:, iternum:iternum + time_steps],
                 model.lng: [time_steps] * batch_size}
            _cost, _, summary = sess.run((model.cost,
                                          model.optimize,
                                          model.merged_summary_op),
                                         feed_dict=d)
            costs.append(_cost)
            model.summary_writer.add_summary(summary, step)
            print("step {}, iteration {}, cost: {}".format(step,
                                                           iternum,
                                                           _cost))
            step = step + 1

            if 'val_error_step' in locals():
                if step % val_error_step == 0:
                    if 'Y_pred_val' in locals():
                        del Y_pred_val

                    for b in range(num_batches_val):  # "b" is "batch number"
                        X_b = X_val_batch[:, b * time_steps: (b + 1) * time_steps, :]
                        Y_b = Y_val_batch[:, b * time_steps: (b + 1) * time_steps]
                        d = {model.X: X_b,
                             model.y: Y_b,
                             model.lng: [time_steps] * batch_size}

                        if 'Y_pred_val' in locals():
                            preds = sess.run(model.predict, feed_dict=d)
                            preds = preds.reshape(batch_size, -1)
                            Y_pred_val = np.concatenate((Y_pred_val, preds), axis=1)
                        else:
                            Y_pred_val = sess.run(model.predict, feed_dict=d)
                            Y_pred_val = Y_pred_val.reshape(batch_size, -1)

                    # get rid of zero padding predictions
                    Y_pred_val = Y_pred_val.ravel()[:Y_val.shape[0], np.newaxis]
                    val_errs.append(np.sum(Y_pred_val - Y_val != 0) / Y_val.shape[0])
                    print("step {}, validation error: {}".format(step, val_errs[-1]))

                if patience:
                    if val_errs[-1] < curr_min_err:
                        # error went down, set as new min and reset counter
                        curr_min_err = val_errs[-1]
                        err_patience_counter = 0
                        checkpoint_path = os.path.join(training_records_path, checkpoint_filename)
                        print("Validation error improved.\n"
                              "Saving checkpoint to {}".format(checkpoint_path))
                        model.saver.save(sess, checkpoint_path)
                    else:
                        err_patience_counter += 1
                        if err_patience_counter > patience:
                            print("stopping because validation error has not improved in {} steps"
                                  .format(patience))
                            with open(os.path.join(training_records_path, "costs"), 'wb') as costs_file:
                                pickle.dump(costs, costs_file)
                            with open(os.path.join(training_records_path, "val_errs"), 'wb') as val_errs_file:
                                pickle.dump(val_errs, val_errs_file)
                            break

            if checkpoint_step:
                if step % checkpoint_step == 0:
                    "Saving checkpoint."
                    checkpoint_path = os.path.join(training_records_path, checkpoint_filename)
                    if save_only_single_checkpoint_file is False:
                        checkpoint_path += '_{}'.format(step)
                    model.saver.save(sess, checkpoint_path)
                    with open(os.path.join(training_records_path, "val_errs"), 'wb') as val_errs_file:
                        pickle.dump(val_errs, val_errs_file)

            if step > n_max_iter:  # ok don't actually loop forever
                "Reached max. number of iterations, saving checkpoint."
                checkpoint_path = os.path.join(training_records_path, checkpoint_filename)
                model.saver.save(sess, checkpoint_path)
                with open(os.path.join(training_records_path, "costs"), 'wb') as costs_file:
                    pickle.dump(costs, costs_file)
                with open(os.path.join(training_records_path, "val_errs"), 'wb') as val_errs_file:
                    pickle.dump(val_errs, val_errs_file)
                break

if __name__ == "__main__":
    config_file = os.path.normpath(sys.argv[1])
    train(config_file)
