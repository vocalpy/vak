import sys
import os
import pickle
from glob import glob
from configparser import ConfigParser

import tensorflow as tf
import numpy as np
from sklearn.externals import joblib

import cnn_bilstm.utils

config_file = sys.argv[1]
if not config_file.endswith('.ini'):
    raise ValueError('{} is not a valid config file, must have .ini extension'
                     .format(config_file))
config = ConfigParser()
config.read(config_file)

results_dirname = config['OUTPUT']['output_dir']
if not os.path.isdir(results_dirname):
    raise FileNotFoundError('{} directory is not found.'
                            .format(results_dirname))

batch_size = int(config['NETWORK']['batch_size'])
time_steps = int(config['NETWORK']['time_steps'])

TRAIN_SET_DURS = [int(element)
                  for element in
                  config['TRAIN']['train_set_durs'].split(',')]
num_replicates = int(config['TRAIN']['replicates'])
REPLICATES = range(num_replicates)
normalize_spectrograms = config.getboolean('DATA', 'normalize_spectrograms')

spect_params = {}
for spect_param_name in ['freq_cutoffs', 'thresh']:
    try:
        if spect_param_name == 'freq_cutoffs':
            freq_cutoffs = [float(element)
                            for element in
                            config['SPECTROGRAM']['freq_cutoffs'].split(',')]
            spect_params['freq_cutoffs'] = freq_cutoffs
        elif spect_param_name == 'thresh':
            spect_params['thresh'] = float(config['SPECTROGRAM']['thresh'])

    except NoOptionError:
        logger.info('Parameter for computing spectrogram, {}, not specified. '
                    'Will use default.'.format(spect_param_name))
        continue
if spect_params == {}:
    spect_params = None

train_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
test_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))

print('loading training data')
labelset = list(config['DATA']['labelset'])
train_data_dir = config['DATA']['data_dir']
number_song_files = int(config['DATA']['number_song_files'])
(train_song_spects,
 train_song_labels,
 timebin_dur) = cnn_bilstm.utils.load_data(labelset,
                                           train_data_dir,
                                           number_song_files,
                                           spect_params)

# reshape training data
X_train = np.concatenate(train_song_spects, axis=0)
Y_train = np.concatenate(train_song_labels, axis=0)
input_vec_size = X_train.shape[-1]

print('loading testing data')
test_data_dir = config['DATA']['test_data_dir']
number_test_song_files = int(config['DATA']['number_test_song_files'])
(test_song_spects,
 test_song_labels) = cnn_bilstm.utils.load_data(labelset,
                                                train_data_dir,
                                                number_test_song_files,
                                                spect_params)[:2]  # [:2] cuz don't need timebin durs again

X_test = np.concatenate(test_song_spects, axis=0)
# copy X_test because it gets scaled and reshape in main loop
X_test_copy = np.copy(X_test)
Y_test = np.concatenate(test_song_labels, axis=0)
# also need copy of Y_test
# because it also gets reshaped in loop
# and because we need to compare with Y_pred
Y_test_copy = np.copy(Y_test)

for dur_ind, train_set_dur in enumerate(TRAIN_SET_DURS):
    for rep_ind, replicate in enumerate(REPLICATES):
        print("getting train and test error for "
              "training set with duration of {} seconds, "
              "replicate {}".format(train_set_dur, replicate))
        training_records_dir = os.path.join(results_dirname,
                                            ('records_for_training_set_with_duration_of_'
                                             + str(train_set_dur) + '_sec_replicate_'
                                             + str(replicate))
                                            )
        checkpoint_filename = ('checkpoint_train_set_dur_'
                               + str(train_set_dur) +
                               '_sec_replicate_'
                               + str(replicate))

        train_inds_file = glob(os.path.join(training_records_dir, 'train_inds'))[0]
        with open(os.path.join(train_inds_file), 'rb') as train_inds_file:
            train_inds = pickle.load(train_inds_file)

        # get training set
        X_train_subset = X_train[train_inds, :]
        Y_train_subset = Y_train[train_inds]
        # normalize before reshaping to avoid even more convoluted array reshaping
        if normalize_spectrograms:
            scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                           .format(train_set_dur, replicate))
            spect_scaler = joblib.load(os.path.join(results_dirname, scaler_name))
            X_train_subset = spect_scaler.transform(X_train_subset)
            X_test = spect_scaler.transform(X_test_copy)
            Y_test = np.copy(Y_test_copy)

        # now that we normalized, we can reshape
        (X_train_subset,
         Y_train_subset,
         num_batches_train) = cnn_bilstm.utils.reshape_data_for_batching(X_train_subset,
                                                                         Y_train_subset,
                                                                         batch_size,
                                                                         time_steps,
                                                                         input_vec_size)
        (X_test,
         Y_test,
         num_batches_test) = cnn_bilstm.utils.reshape_data_for_batching(X_test,
                                                                        Y_test,
                                                                        batch_size,
                                                                        time_steps,
                                                                        input_vec_size)

        meta_file = glob(os.path.join(training_records_dir, 'checkpoint*meta*'))[0]
        data_file = glob(os.path.join(training_records_dir, 'checkpoint*data*'))[0]

        with tf.Session(graph=tf.Graph()) as sess:
            tf.logging.set_verbosity(tf.logging.ERROR)
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, data_file[:-20])

            # Retrieve the Ops we 'remembered'.
            logits = tf.get_collection("logits")[0]
            X = tf.get_collection("specs")[0]
            Y = tf.get_collection("labels")[0]
            lng = tf.get_collection("lng")[0]

            # Add an Op that chooses the top k predictions.
            eval_op = tf.nn.top_k(logits)

            if 'Y_pred' in locals():
                del Y_pred

            print('calculating training set error')
            for b in range(num_batches_train):  # "b" is "batch number"
                d = {X: X_train_subset[:, b * time_steps: (b + 1) * time_steps, :],
                     Y: Y_train_subset[:, b * time_steps: (b + 1) * time_steps],
                     lng: [time_steps] * batch_size}

                if 'Y_pred' in locals():
                    preds = sess.run(eval_op, feed_dict=d)[1]
                    preds = preds.reshape(batch_size, -1)
                    Y_pred = np.concatenate((Y_pred, preds), axis=1)
                else:
                    Y_pred = sess.run(eval_op, feed_dict=d)[1]
                    Y_pred = Y_pred.reshape(batch_size, -1)

            Y_train_arr = Y_train[train_inds]
            Y_pred = Y_pred.ravel()[:Y_train_arr.shape[0], np.newaxis]
            train_err = np.sum(Y_pred - Y_train_arr != 0) / Y_train_arr.shape[0]
            train_err_arr[dur_ind, rep_ind] = train_err
            print('train error was {}'.format(train_err))

            if 'Y_pred' in locals():
                del Y_pred

            print('calculating test set error')
            for b in range(num_batches_test):  # "b" is "batch number"
                d = {X: X_test[:, b * time_steps: (b + 1) * time_steps, :],
                     Y: Y_test[:, b * time_steps: (b + 1) * time_steps],
                     lng: [time_steps] * batch_size}

                if 'Y_pred' in locals():
                    preds = sess.run(eval_op, feed_dict=d)[1]
                    preds = preds.reshape(batch_size, -1)
                    Y_pred = np.concatenate((Y_pred, preds), axis=1)
                else:
                    Y_pred = sess.run(eval_op, feed_dict=d)[1]
                    Y_pred = Y_pred.reshape(batch_size, -1)

            Y_pred = Y_pred.ravel()[:Y_test_copy.shape[0], np.newaxis]
            test_err = np.sum(Y_pred - Y_test_copy != 0) / Y_test_copy.shape[0]
            test_err_arr[dur_ind, rep_ind] = test_err
            print('test error was {}'.format(test_err))

train_err_filename = os.path.join(training_records_dir,
                                  'train_err')
with open(train_err_filename,'wb') as train_err_file:
    pickle.dump(train_err_arr, train_err_file)

test_err_filename = os.path.join(training_records_dir,
                                  'test_err')
with open(test_err_filename, 'wb') as test_err_file:
    pickle.dump(test_err_arr, test_err_file)
