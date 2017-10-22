import os
import pickle
from glob import glob

import tensorflow as tf
import numpy as np

import cnn_bilstm.utils


def reshape_data_for_batching(X, Y, batch_size, time_steps, input_vec_size):
    """reshape to feed to network in batches"""
    # need to loop through train data in chunks, can't fit on GPU all at once
    # First zero pad
    num_batches = X.shape[0] // batch_size // time_steps
    rows_to_append = ((num_batches + 1) * time_steps * batch_size) - X.shape[0]
    X = np.append(X, np.zeros((rows_to_append, input_vec_size)),
                               axis=0)
    Y = np.append(Y, np.zeros((rows_to_append, 1)), axis=0)
    num_batches = num_batches + 1
    X = X.reshape((batch_size, num_batches * time_steps, -1))
    Y = Y.reshape((batch_size, -1))
    return X, Y, num_batches

batch_size = 11
time_steps = 87
input_vec_size = 513

TRAIN_SET_DURS = [5, 15, 30, 45, 60, 75, 90, 105, 120]
REPLICATES = list(range(5))
train_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
test_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))

print('loading training data')
labelset = list('iabcdefghjk')
train_data_dir = 'C:\\DATA\\gy6or6\\032212\\'
(train_song_spects,
 train_song_labels,
 timebin_dur) = cnn_bilstm.utils.load_data(labelset,
                                           train_data_dir,
                                           number_files=20)
# reshape training data
X_train = np.concatenate(train_song_spects, axis=0)
Y_train = np.concatenate(train_song_labels, axis=0)

print('loading testing data')
test_data_dir = 'C:\\DATA\\gy6or6\\032312\\'
(test_song_spects,
 test_song_labels) = cnn_bilstm.utils.load_data(labelset,
                                                train_data_dir,
                                                number_files=20)[:2]  # don't need timebin durs again
X_test = np.concatenate(test_song_spects, axis=0)
Y_test = np.concatenate(test_song_labels, axis=0)
Y_test_arr = Y_test  # for comparing with predictions below
(X_test,
 Y_test,
 num_batches_test) = reshape_data_for_batching(X_test,
                                               Y_test,
                                               batch_size,
                                               time_steps,
                                               input_vec_size)

results_dirname = 'C:\\workspace\\tf_syl_seg_fork\\results_171021_164437'

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
         num_batches_train) = reshape_data_for_batching(X_train_subset,
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
                    Y_pred = np.concatenate((Y_pred,preds), axis=1)
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
                    Y_pred = np.concatenate((Y_pred,preds), axis=1)
                else:
                    Y_pred = sess.run(eval_op, feed_dict=d)[1]
                    Y_pred = Y_pred.reshape(batch_size, -1)

            Y_pred = Y_pred.ravel()[:Y_test_arr.shape[0], np.newaxis]
            test_err = np.sum(Y_pred - Y_test_arr != 0) / Y_test_arr.shape[0]
            test_err_arr[dur_ind, rep_ind] = test_err
            print('test error was {}'.format(test_err))

with open('train_err','wb') as train_err_file:
    pickle.dump(train_err_arr, train_err_file)

with open('train_err', 'wb') as test_err_file:
    pickle.dump(test_err_arr, test_err_file)
