import sys
import os
import pickle
from glob import glob
from configparser import ConfigParser

import tensorflow as tf
import numpy as np

import cnn_bilstm.utils

if len(sys.argv[1:]) != 2:
    raise ValueError('there should be two arguments to this function, '
                     'the first being the .ini file and the second being '
                     'the results dir. Instead got {} arguments: {}'
                     .format(len(sys.argv[1:]), sys.argv[1:]))

if not os.path.isdir(sys.argv[2]):
    raise FileNotFoundError('{} directory is not found.'
                            .format(sys.argv[2]))

config_file = sys.argv[1]
if not config_file.endswith('.ini'):
    raise ValueError('{} is not a valid config file, must have .ini extension'
                     .format(config_file))
config = ConfigParser()
config.read(config_file)

batch_size = int(config['NETWORK']['batch_size'])
time_steps = int(config['NETWORK']['time_steps'])
input_vec_size = int(config['NETWORK']['input_vec_size'])

TRAIN_SET_DURS = [int(element)
                  for element in
                  config['TRAIN']['train_set_durs'].split(',')]
num_replicates = int(config['TRAIN']['replicates'])
REPLICATES = range(num_replicates)

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
                                           number_song_files)

# reshape training data
X_train = np.concatenate(train_song_spects, axis=0)
Y_train = np.concatenate(train_song_labels, axis=0)

print('loading testing data')
test_data_dir = config['DATA']['test_data_dir']
number_test_song_files = int(config['DATA']['number_test_song_files'])
(test_song_spects,
 test_song_labels) = cnn_bilstm.utils.load_data(labelset,
                                                train_data_dir,
                                                number_test_song_files)[:2]  # [:2] cuz don't need timebin durs again

X_test = np.concatenate(test_song_spects, axis=0)
Y_test = np.concatenate(test_song_labels, axis=0)
Y_test_arr = Y_test  # for comparing with predictions below
(X_test,
 Y_test,
 num_batches_test) = cnn_bilstm.utils.reshape_data_for_batching(X_test,
                                                                Y_test,
                                                                batch_size,
                                                                time_steps,
                                                                input_vec_size)

results_dirname = sys.argv[2]

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
        (X_train_subset,
         Y_train_subset,
         num_batches_train) = cnn_bilstm.utils.reshape_data_for_batching(X_train_subset,
                                                                         Y_train_subset,
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

            Y_pred = Y_pred.ravel()[:Y_test_arr.shape[0], np.newaxis]
            test_err = np.sum(Y_pred - Y_test_arr != 0) / Y_test_arr.shape[0]
            test_err_arr[dur_ind, rep_ind] = test_err
            print('test error was {}'.format(test_err))

with open('train_err','wb') as train_err_file:
    pickle.dump(train_err_arr, train_err_file)

with open('test_err', 'wb') as test_err_file:
    pickle.dump(test_err_arr, test_err_file)
