import sys
import os
import logging
import pickle
from datetime import datetime
from configparser import ConfigParser

import tensorflow as tf
import numpy as np
from sklearn.externals import joblib

from cnn_bilstm.graphs import get_full_graph
import cnn_bilstm.utils

if __name__ == "__main__":
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    results_dirname = os.path.join('.', 'results_' + timenow)
    os.mkdir(results_dirname)

    logfile_name = os.path.join(results_dirname,
                                'metadata_from_running_main_' + timenow + '.log')
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
    logger.addHandler(logging.FileHandler(logfile_name))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    config_file = sys.argv[1]
    if not config_file.endswith('.ini'):
        raise ValueError('{} is not a valid config file, must have .ini extension'
                         .format(config_file))
    config = ConfigParser()
    config.read(config_file)
    logger.info('Using config file: {}'.format(config_file))

    print('loading data for training')
    labelset = list(config['DATA']['labelset'])
    data_dir = config['DATA']['data_dir']
    number_song_files = int(config['DATA']['number_song_files'])
    song_spects, all_labels, timebin_dur = cnn_bilstm.utils.load_data(labelset,
                                                                      data_dir,
                                                                      number_song_files)
    logger.info('Loading training data from {}'.format(data_dir))
    logger.info('Using first {} songs'.format(number_song_files))

    # note that training songs are taken from the start of the training data
    # and validation songs are taken starting from the end

    # reshape training data
    num_train_songs = int(config['DATA']['num_train_songs'])
    train_spects = song_spects[:num_train_songs]

    X_train = np.concatenate(train_spects, axis=0)
    normalize_spectrograms = config.getboolean('DATA', 'normalize_spectrograms')
    if normalize_spectrograms:
        logger.info('normalizing spectrograms')
        spect_scaler = cnn_bilstm.utils.SpectScaler()
        X_train = spect_scaler.fit_transform(X_train)
        joblib.dump(spect_scaler, os.path.join(results_dirname, 'spect_scaler'))

    joblib.dump(X_train, os.path.join(results_dirname, 'X_train'))
    X_train_durations = [spec.shape[0] for spec in train_spects]  # rows are time bins
    Y_train = np.concatenate(all_labels[:num_train_songs], axis=0)
    total_train_set_duration = sum(X_train_durations) / 1000
    logger.info('Total duration of training set (in s): {}'
                .format(total_train_set_duration))

    TRAIN_SET_DURS = [int(element)
                      for element in
                      config['TRAIN']['train_set_durs'].split(',')]
    logger.info('Will train network with training sets of '
                'following durations (in s): {}'.format(TRAIN_SET_DURS))
    max_train_set_dur = np.max(TRAIN_SET_DURS)
    if max_train_set_dur > total_train_set_duration:
        raise ValueError('Largest duration for a training set of {} '
                         'is greater than total duration of training set, {}'
                         .format(max_train_set_dur, total_train_set_duration))

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
                         'in the validation set. Please increaes the number '
                         'of songfiles or decrease the size of the training '
                         'or validation set.'
                         .format(num_train_songs,
                                 num_val_songs,
                                 num_train_songs + num_val_songs,
                                 number_song_files))
    X_val = song_spects[-num_val_songs:]
    if normalize_spectrograms:
        logger.info('normalizing validation set to match training set')
        X_val = spect_scaler.transform(X_val)
    joblib.dump(X_val, os.path.join(results_dirname, 'X_val'))
    Y_val = all_labels[-num_val_songs:]

    Y_val_arr = np.concatenate(Y_val, axis=0)

    val_error_step = int(config['TRAIN']['val_error_step'])
    logger.info('will measure error on validation set '
                'every {} steps of training'.format(val_error_step))
    checkpoint_step = int(config['TRAIN']['checkpoint_step'])
    logger.info('will save a checkpoint file '
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
            train_inds = cnn_bilstm.utils.get_inds_for_dur(X_train_durations,
                                                           train_set_dur,
                                                           timebin_dur)
            with open(os.path.join(training_records_dir, 'train_inds'),
                      'wb') as train_inds_file:
                pickle.dump(train_inds, train_inds_file)
            X_train_subset = X_train[train_inds, :]
            Y_train_subset = Y_train[train_inds]

            batch_spec_rows = len(train_inds) // batch_size
            X_train_subset = \
                X_train_subset[0:batch_spec_rows * batch_size].reshape((batch_size,
                                                                        batch_spec_rows,
                                                                        -1))
            Y_train_subset = \
                Y_train_subset[0:batch_spec_rows * batch_size].reshape((batch_size,
                                                                        -1))
            iter_order = np.random.permutation(X_train.shape[1] - time_steps)
            if len(iter_order) > n_max_iter:
                iter_order = iter_order[0:n_max_iter]

            input_vec_size = int(config['NETWORK']['input_vec_size'])
            logger.debug('input vec size: '.format(input_vec_size))
            num_hidden = int(config['NETWORK']['num_hidden'])
            logger.debug('num_hidden: '.format(num_hidden))
            n_syllables = int(config['NETWORK']['n_syllables'])
            logger.debug('n_syllables: '.format(n_syllables))
            learning_rate = float(config['NETWORK']['learning_rate'])
            logger.debug('learning rate: '.format(learning_rate))

            logger.debug('creating graph')
            (full_graph, train_op, cost,
             init, saver, logits, X, Y, lng) = get_full_graph(input_vec_size,
                                                              num_hidden,
                                                              n_syllables,
                                                              learning_rate,
                                                              batch_size)
            # Add an Op that chooses the top k predictions.
            eval_op = tf.nn.top_k(logits)

            with tf.Session(graph=full_graph,
                            config=tf.ConfigProto(
                                intra_op_parallelism_threads=512)
                            ) as sess:
                # ,config = tf.ConfigProto(intra_op_parallelism_threads = 1)
                # Run the Op to initialize the variables.
                sess.run(init)
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
                    _cost, _ = sess.run((cost, train_op), feed_dict=d)
                    costs.append(_cost)
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

                            d = {X: X_val_song_reshape,
                                 Y: Y_val_song_reshape,
                                 lng: [temp_n] * batch_size
                                 }

                            unpadded_length = Y_val_song.shape[0]

                            if 'preds' in locals():
                                preds = np.concatenate((preds,
                                                        sess.run(eval_op, feed_dict=d)[1][:unpadded_length]))
                            else:
                                preds = sess.run(eval_op, feed_dict=d)[1][:unpadded_length]  # eval_op

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
                            saver.save(sess, checkpoint_path)
                            with open(os.path.join(training_records_dir, "costs"), 'wb') as costs_file:
                                pickle.dump(costs, costs_file)
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


