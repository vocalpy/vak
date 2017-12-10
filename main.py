import sys
import os
import shutil
import copy
import logging
import pickle
from datetime import datetime
from configparser import ConfigParser, NoOptionError

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import joblib

from cnn_bilstm.graphs import get_full_graph
from cnn_bilstm.utils import make_data_dict, reshape_data_for_batching
import cnn_bilstm.utils

if __name__ == "__main__":
    config_file = sys.argv[1]
    if not config_file.endswith('.ini'):
        raise ValueError('{} is not a valid config file, '
                         'must have .ini extension'.format(config_file))
    config = ConfigParser()
    config.read(config_file)

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    if config.has_section('OUTPUT'):
        if config.has_option('OUTPUT', 'results_dir'):
            output_dir = config['OUTPUT']['results_dir']
            results_dirname = os.path.join(output_dir,
                                           'results_' + timenow)
    else:
        results_dirname = os.path.join('.', 'results_' + timenow)
    os.mkdir(results_dirname)
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
    spect_params['thresh'] = float(config['SPECTROGRAM']['thresh'])
    spect_params['log_transform'] = config.getboolean('SPECTROGRAM',
                                                      'log_transform')

    # need to know number_song_files regardless of
    # whether we make data or load from file
    number_song_files = int(config['DATA']['number_song_files'])
    make_train_data = config.getboolean('DATA', 'make_train_data')

    if make_train_data:
        logger.info('make_train_data = Yes')
        train_data_dir = config['DATA']['train_data_dir']
        logger.info('will make training data from: {}'.format(train_data_dir))

        labelset = list(config['DATA']['labelset'])
        # make mapping from syllable labels to consecutive integers
        # start at 1, because 0 is assumed to be label for silent gaps
        labels_mapping = dict(zip(labelset,
                                  range(1, len(labelset) + 1)))
        skip_files_with_labels_not_in_labelset = config.getboolean(
            'DATA',
            'skip_files_with_labels_not_in_labelset')
        logger.info('Using first {} songs'.format(number_song_files))


        (train_data_dict,
         train_data_dict_path) = make_data_dict(labels_mapping,
                                                train_data_dir,
                                                number_song_files,
                                                spect_params,
                                                skip_files_with_labels_not_in_labelset)
        logger.info('saved training data as {}'.format(train_data_dict_path))
    else:
        if config.has_option('DATA', 'train_data_path'):
            train_data_dict_path = config['DATA']['train_data_path']
            logger.info('Loading training data from {}'.format(
                os.path.dirname(
                    train_data_dict_path)))
        elif config.has_option('DATA', 'train_data_dir'):
            train_data_dir = config['DATA']['train_data_dir']
            train_data_dict_path = os.path.join(train_data_dir,
                                               'data_dict')
            logger.info('Loading training data from {}'.format(train_data_dir))
        train_data_dict = joblib.load(train_data_dict_path)
        labels_mapping = train_data_dict['labels_mapping']

        if train_data_dict['spect_params'] is None:
            logger.warning('spect_params is None in data_dict loaded,'
                           'probably because spectrograms were generated'
                           ' in Matlab.\nNot checking whether parameters '
                           'match those specified in .ini file.')
        else:
            if train_data_dict['spect_params'] != spect_params:
                raise ValueError('Spectrogram parameters in {} '
                                 'do not match parameters specified in data_dict '
                                 'from {}.'.format(train_data_dict_path,
                                                   train_data_dir))

    # save copy of labels_mapping in results directory
    labels_mapping_file = os.path.join(results_dirname, 'labels_mapping')
    with open(labels_mapping_file, 'wb') as labels_map_file_obj:
        pickle.dump(labels_mapping, labels_map_file_obj)

    # copy training data to results dir so we have it stored with results
    logger.info('copying {} to {}'.format(train_data_dict_path,
                                          results_dirname))
    # rename to 'train_data_dict' so we don't write over with 'data_dict' from testing
    train_data_copy_filename = os.path.join(results_dirname,
                                            'train_data_dict')
    shutil.copy(train_data_dict_path, train_data_copy_filename)

    (train_data_spects,
     train_labeled_timebins,
     timebin_dur,
     files_used) = (train_data_dict['spects'],
                    train_data_dict['labeled_timebins'],
                    train_data_dict['timebin_dur'],
                    train_data_dict['filenames'])

    logger.info('Size of each timebin in spectrogram, in seconds: {}'
                .format(timebin_dur))
    # dump filenames to a text file
    # to be consistent with what the matlab helper function does
    files_used_filename = os.path.join(results_dirname, 'training_filenames')
    with open(files_used_filename, 'w') as files_used_fileobj:
        files_used_fileobj.write('\n'.join(files_used))

    # reshape training data
    num_train_songs = int(config['DATA']['num_train_songs'])
    # note that training songs are taken from the start of the training data
    # and validation songs are taken starting from the end
    spects_for_X_train = train_data_spects[:num_train_songs]
    #####################################################
    # note that we 'transpose' the spectrogram          #
    # so that rows are time and columns are frequencies #
    #####################################################
    spects_for_X_train = [spect.T for spect in spects_for_X_train]
    X_train_timebins = np.array(
        [spec.shape[0] for spec in spects_for_X_train]
        # spec.shape[0] because rows are time bins
    )  # convert to seconds by multiplying by size of time bin
    X_train_durations = X_train_timebins * timebin_dur
    total_train_set_duration = sum(X_train_durations)
    logger.info('Total duration of training set (in s): {}'
                .format(total_train_set_duration))
    TRAIN_SET_DURS = [int(element)
                      for element in
                      config['TRAIN']['train_set_durs'].split(',')]
    max_train_set_dur = np.max(TRAIN_SET_DURS)

    if max_train_set_dur > total_train_set_duration:
        raise ValueError('Largest duration for a training set of {} '
                         'is greater than total duration of training set, {}'
                         .format(max_train_set_dur, total_train_set_duration))

    logger.info('Will train network with training sets of '
                'following durations (in s): {}'.format(TRAIN_SET_DURS))

    X_train = np.concatenate(spects_for_X_train, axis=0)
    # save training set to get training accuracy in summary.py
    joblib.dump(X_train, os.path.join(results_dirname, 'X_train'))
    Y_train = np.concatenate(train_labeled_timebins[:num_train_songs], axis=0)
    joblib.dump(Y_train, os.path.join(results_dirname, 'Y_train'))

    num_replicates = int(config['TRAIN']['replicates'])
    REPLICATES = range(num_replicates)
    logger.info('will replicate training {} times for each duration of training set'
                .format(num_replicates))

    num_val_songs = int(config['DATA']['num_val_songs'])
    logger.info('validation set used during training will contain {} songs'
                .format(num_val_songs))
    if num_train_songs + num_val_songs > number_song_files:
        raise ValueError('Total number of training songs ({0}), '
                         'and validation songs ({1}), '
                         'is {2}.\n This is greater than the number of '
                         'songfiles, {3}, and would result in training data '
                         'in the validation set. Please increase the number '
                         'of songfiles or decrease the size of the training '
                         'or validation set.'
                         .format(num_train_songs,
                                 num_val_songs,
                                 num_train_songs + num_val_songs,
                                 number_song_files))
    X_val = train_data_spects[-num_val_songs:]
    #####################################################
    # note that we 'transpose' the spectrogram          #
    # so that rows are time and columns are frequencies #
    #####################################################
    X_val = [spect.T for spect in X_val]
    joblib.dump(X_val, os.path.join(results_dirname, 'X_val'))
    X_val_copy = copy.deepcopy(X_val)  # need a copy if we scale X_val below
    Y_val = train_labeled_timebins[-num_val_songs:]
    joblib.dump(X_val, os.path.join(results_dirname, 'Y_val'))
    Y_val_arr = np.concatenate(Y_val, axis=0)

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

    normalize_spectrograms = config.getboolean('DATA', 'normalize_spectrograms')
    if normalize_spectrograms:
        logger.info('will normalize spectrograms for each training set')

    for train_set_dur in TRAIN_SET_DURS:
        for replicate in REPLICATES:
            costs = []
            val_errs = []
            curr_min_err = 1  # i.e. 100%
            err_patience_counter = 0

            logger.info("training with training set duration of {} seconds,"
                        "replicate #{}".format(train_set_dur, replicate))
            training_records_dir = os.path.join(results_dirname,
                ('records_for_training_set_with_duration_of_'
                                    + str(train_set_dur) + '_sec_replicate_'
                                    + str(replicate))
                                                )
            checkpoint_filename = ('checkpoint_train_set_dur_'
                                   + str(train_set_dur) +
                                   '_sec_replicate_'
                                   + str(replicate))
            if not os.path.isdir(training_records_dir):
                os.mkdir(training_records_dir)
            train_inds = cnn_bilstm.utils.get_inds_for_dur(X_train_timebins,
                                                           train_set_dur,
                                                           timebin_dur)
            with open(os.path.join(training_records_dir, 'train_inds'),
                      'wb') as train_inds_file:
                pickle.dump(train_inds, train_inds_file)
            X_train_subset = X_train[train_inds, :]
            Y_train_subset = Y_train[train_inds]

            if normalize_spectrograms:
                spect_scaler = cnn_bilstm.utils.SpectScaler()
                X_train_subset = spect_scaler.fit_transform(X_train_subset)
                logger.info('normalizing validation set to match training set')
                X_val = spect_scaler.transform(X_val_copy)
                scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                               .format(train_set_dur, replicate))
                joblib.dump(spect_scaler,
                            os.path.join(results_dirname, scaler_name))

            scaled_data_filename = os.path.join(training_records_dir,
                                                'scaled_spects_duration_{}_replicate_{}'
                                                .format(train_set_dur, replicate))
            scaled_data_dict = {'X_train_subset_scaled': X_train_subset,
                                'X_val_scaled': X_val}
            joblib.dump(scaled_data_dict, scaled_data_filename)

            # reshape data for network
            batch_spec_rows = len(train_inds) // batch_size

            # not sure of best way to reshape without losing a few data bins
            # or adding unnecessary 'recycled' time bins or zero padding
            # (X_train_subset,
            #  Y_train_subset,
            #  num_batches_train) = reshape_data_for_batching(X_train_subset,
            #                                                 Y_train_subset,
            #                                                 batch_size,
            #                                                 time_steps,
            #                                                 input_vec_size)

            # this is the original way reshaping was done
            # note that reshaping this way can truncate data set
            X_train_subset = \
                X_train_subset[0:batch_spec_rows * batch_size].reshape((batch_size,
                                                                        batch_spec_rows,
                                                                        -1))
            Y_train_subset = \
                Y_train_subset[0:batch_spec_rows * batch_size].reshape((batch_size,
                                                                        -1))
            reshape_size = Y_train_subset.ravel().shape[-1]
            diff = train_inds.shape[-1] - reshape_size
            logger.info('Number of time bins after '
                        'reshaping training data: {}.'.format(reshape_size))
            logger.info('Number of time bins less '
                        'than specified {}: {}'.format(train_inds.shape[-1],
                                                       diff))
            logger.info('Difference in seconds: {}'.format(diff * timebin_dur))

            # save scaled reshaped data
            scaled_reshaped_data_filename = os.path.join(training_records_dir,
                                                         'scaled_reshaped_spects_duration_{}_replicate_{}'
                                                         .format(train_set_dur, replicate))
            scaled_reshaped_data_dict = {'X_train_subset_scaled_reshaped': X_train_subset,
                                         'Y_train_subset_reshaped': Y_train_subset}
            joblib.dump(scaled_reshaped_data_dict, scaled_reshaped_data_filename)

            # note that X_train_subset has shape of (batch, time_bins, frequency_bins)
            # so we permute starting indices from the number of time_bins
            # i.e. X_train_subset.shape[1]
            iter_order = np.random.permutation(X_train_subset.shape[1] - time_steps)
            if len(iter_order) > n_max_iter:
                iter_order = iter_order[0:n_max_iter]
            with open(
                    os.path.join(training_records_dir,
                                 "iter_order"),
                    'wb') as iter_order_file:
                pickle.dump(iter_order, iter_order_file)

            input_vec_size = X_train_subset.shape[-1]  # number of columns
            logger.debug('input vec size: '.format(input_vec_size))
            num_hidden = int(config['NETWORK']['num_hidden'])
            logger.debug('num_hidden: '.format(num_hidden))
            # n_syllables, i.e., number of label classes to predict
            # is length of labels mapping plus one for "silent gap"
            # class
            n_syllables = len(labels_mapping) + 1
            logger.debug('n_syllables: '.format(n_syllables))
            learning_rate = float(config['NETWORK']['learning_rate'])
            logger.debug('learning rate: '.format(learning_rate))

            logger.debug('creating graph')

            (full_graph, train_op, cost,
             init, saver, logits, X, Y, lng,
             merged_summary_op) = get_full_graph(input_vec_size,
                                                 num_hidden,
                                                 n_syllables,
                                                 learning_rate,
                                                 batch_size)

            # Add an Op that chooses the top k predictions.
            eval_op = tf.nn.top_k(logits)

            logs_path = os.path.join(results_dirname,'logs')
            if not os.path.isdir(logs_path):
                os.mkdir(logs_path)

            with tf.Session(graph=full_graph,
                            config=tf.ConfigProto(
                                log_device_placement=True
                                # intra_op_parallelism_threads=512
                            )) as sess:

                # Run the Op to initialize the variables.
                sess.run(init)

                # op to write logs to Tensorboard
                summary_writer = tf.summary.FileWriter(logs_path,
                                                       graph=tf.get_default_graph())

                if '--debug' in sys.argv:
                    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

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
                    d = {X: X_train_subset[:, iternum:iternum + time_steps, :],
                         Y: Y_train_subset[:, iternum:iternum + time_steps],
                         lng: [time_steps] * batch_size}
                    _cost, _, summary = sess.run((cost, train_op, merged_summary_op),
                                        feed_dict=d)
                    costs.append(_cost)
                    summary_writer.add_summary(summary, step)
                    print("step {}, iteration {}, cost: {}".format(step, iternum, _cost))
                    step = step + 1

                    if step % val_error_step == 0:
                        if 'preds' in locals():
                            del preds

                        for X_val_song, Y_val_song in zip(X_val, Y_val):
                            temp_n = len(Y_val_song) // batch_size
                            rows_to_append = (temp_n + 1) * batch_size - X_val_song.shape[0]
                            X_val_song_padded = np.append(X_val_song, np.zeros((rows_to_append, input_vec_size)), axis=0)
                            Y_val_song_padded = np.append(Y_val_song, np.zeros((rows_to_append, 1)), axis=0)
                            temp_n = temp_n + 1
                            X_val_song_reshape = X_val_song_padded[0:temp_n * batch_size].reshape((batch_size, temp_n, -1))
                            Y_val_song_reshape = Y_val_song_padded[0:temp_n * batch_size].reshape((batch_size, -1))

                            d_val = {X: X_val_song_reshape,
                                     Y: Y_val_song_reshape,
                                     lng: [temp_n] * batch_size
                                     }

                            unpadded_length = Y_val_song.shape[0]

                            if 'preds' in locals():
                                preds = np.concatenate((preds,
                                                        sess.run(eval_op, feed_dict=d_val)[1][:unpadded_length]))
                            else:
                                preds = sess.run(eval_op, feed_dict=d_val)[1][:unpadded_length]  # eval_op

                        val_errs.append(np.sum(preds - Y_val_arr != 0) / Y_val_arr.shape[0])
                        print("step {}, validation error: {}".format(step, val_errs[-1]))

                        if patience:
                            if val_errs[-1] < curr_min_err:
                                # error went down, set as new min and reset counter
                                curr_min_err = val_errs[-1]
                                err_patience_counter = 0
                                checkpoint_path = os.path.join(training_records_dir, checkpoint_filename)
                                print("Validation error improved.\n"
                                      "Saving checkpoint to {}".format(checkpoint_path))
                                saver.save(sess, checkpoint_path)
                            else:
                                err_patience_counter += 1
                                if err_patience_counter > patience:
                                    print("stopping because validation error has not improved in {} steps"
                                          .format(patience))
                                    with open(os.path.join(training_records_dir, "costs"), 'wb') as costs_file:
                                        pickle.dump(costs, costs_file)
                                    with open(os.path.join(training_records_dir, "val_errs"), 'wb') as val_errs_file:
                                        pickle.dump(val_errs, val_errs_file)
                                    break

                    if checkpoint_step:
                        if step % checkpoint_step == 0:
                            "Saving checkpoint."
                            checkpoint_path = os.path.join(training_records_dir, checkpoint_filename)
                            if save_only_single_checkpoint_file is False:
                                checkpoint_path += '_{}'.format(step)
                            saver.save(sess, checkpoint_path)
                            with open(os.path.join(training_records_dir, "val_errs"), 'wb') as val_errs_file:
                                pickle.dump(val_errs, val_errs_file)

                    if step > n_max_iter:  # ok don't actually loop forever
                        "Reached max. number of iterations, saving checkpoint."
                        checkpoint_path = os.path.join(training_records_dir, checkpoint_filename)
                        saver.save(sess, checkpoint_path)
                        with open(os.path.join(training_records_dir, "costs"), 'wb') as costs_file:
                            pickle.dump(costs, costs_file)
                        with open(os.path.join(training_records_dir, "val_errs"), 'wb') as val_errs_file:
                            pickle.dump(val_errs, val_errs_file)
                        break
