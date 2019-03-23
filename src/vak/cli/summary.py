import os
import pickle
import sys
from datetime import datetime
from glob import glob

import joblib
import numpy as np
import tensorflow as tf

from .. import metrics, utils
import vak.network


def summary(results_dirname,
            train_data_dict_path,
            networks,
            train_set_durs,
            num_replicates,
            labelset,
            test_data_dict_path,
            normalize_spectrograms=False):
    """generate summary learning curve from networks trained by cli.learncurve

    Parameters
    ----------
    results_dirname
    train_data_dict_path
    networks
    train_set_durs
    num_replicates
    labelset
    test_data_dict_path
    normalize_spectrograms

    Returns
    -------
    None

    Computes error on test set and saves in a ./summary directory within results directory
    """
    if not os.path.isdir(results_dirname):
        raise FileNotFoundError('directory {}, specified as '
                                'results_dir_made_by_main_script, is not found.'
                                .format(results_dirname))
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    summary_dirname = os.path.join(results_dirname,
                                   'summary_' + timenow)
    os.makedirs(summary_dirname)

    labels_mapping_file = os.path.join(results_dirname, 'labels_mapping')
    with open(labels_mapping_file, 'rb') as labels_map_file_obj:
        labels_mapping = pickle.load(labels_map_file_obj)

    train_data_dict = joblib.load(train_data_dict_path)
    (X_train,
     Y_train,
     train_timebin_dur,
     train_spect_params,
     train_labels) = (train_data_dict['X_train'],
                      train_data_dict['Y_train'],
                      train_data_dict['timebin_dur'],
                      train_data_dict['spect_params'],
                      train_data_dict['labels'])
    labels_mapping = train_data_dict['labels_mapping']
    n_syllables = len(labels_mapping)
    X_train = X_train.T

    # only get this just to have in summary file if needed
    if all(type(labels_el) is str for labels_el in train_labels):
        # when taken from .not.mat files associated with .cbin audio files
        Y_train_labels = ''.join(train_labels)
        Y_train_labels_for_lev = Y_train_labels
    elif all(type(labels_el) is np.ndarray for labels_el in train_labels):
        # when taken from annotation.mat supplied with .wav audio files
        Y_train_labels = np.concatenate(train_labels).tolist()
        Y_train_labels_for_lev = ''.join([chr(lbl) for lbl in Y_train_labels])
    elif all(type(labels_el) is list for labels_el in train_labels):
        # when taken from annotation.xml supplied with Koumura .wav audio files
        Y_train_labels = [lbl for lbl_list in train_labels for lbl in lbl_list]
        if all([type(lbl) is int for lbl in Y_train_labels]):
            Y_train_labels_for_lev = ''.join(
                [chr(lbl) for lbl in Y_train_labels])
        elif all([type(lbl) is str for lbl in Y_train_labels]):
            Y_train_labels_for_lev = ''.join(Y_train_labels)
        else:
            raise TypeError('Couldn\'t determine type for training labels in {}'
                            .format(type(train_data_dict_path)))
    else:
        raise TypeError('Not able to determine type of labels in train data')

    print('loading testing data')
    test_data_dict = joblib.load(test_data_dict_path)

    # notice data is called `X_test_copy` and `Y_test_copy`
    # because main loop below needs a copy of the original
    # to normalize and reshape
    (X_test_copy,
     Y_test_copy,
     test_timebin_dur,
     files_used,
     test_spect_params,
     test_labels) = (test_data_dict['X_test'],
                     test_data_dict['Y_test'],
                     test_data_dict['timebin_dur'],
                     test_data_dict['filenames'],
                     test_data_dict['spect_params'],
                     test_data_dict['labels'])

    if train_spect_params != test_spect_params:
        raise ValueError('Spectrogram parameters for training data do not match those '
                         'for test data, will give incorrect error rate.')
    if train_timebin_dur != test_timebin_dur:
        raise ValueError('Durations of time bins in spectrograms for training data '
                         'does not match that of test data, will give incorrect error rate.')

    # have to transpose X_test so rows are timebins and columns are frequencies
    X_test_copy = X_test_copy.T

    # save test set so it's clear from results directory alone
    # which test set was used
    joblib.dump(X_test_copy, os.path.join(results_dirname, 'X_test'))
    joblib.dump(Y_test_copy, os.path.join(results_dirname, 'Y_test'))

    # used for Levenshtein distance + syllable error rate
    if all(type(labels_el) is str for labels_el in test_labels):
        # when taken from .not.mat files associated with .cbin audio files
        Y_test_labels = ''.join(test_labels)
        Y_test_labels_for_lev = Y_test_labels
    elif all(type(labels_el) is np.ndarray for labels_el in test_labels):
        # when taken from annotation.mat supplied with .wav audio files
        Y_test_labels = np.concatenate(test_labels).tolist()
        Y_test_labels_for_lev = ''.join([chr(lbl) for lbl in Y_test_labels])
    elif all(type(labels_el) is list for labels_el in test_labels):
        # when taken from annotation.xml supplied with Koumura .wav audio files
        Y_test_labels = [lbl for lbl_list in test_labels for lbl in lbl_list]
        if all([type(lbl) is int for lbl in Y_train_labels]):
            Y_test_labels_for_lev = ''.join(
                [chr(lbl) for lbl in Y_test_labels])
        elif all([type(lbl) is str for lbl in Y_train_labels]):
            Y_test_labels_for_lev = ''.join(Y_test_labels)
        else:
            raise TypeError('Couldn\'t determine type for test labels in {}'
                            .format(type(test_data_dict_path)))
    else:
        raise TypeError('Not able to determine type of labels in test data')

    replicates = range(num_replicates)
    # initialize arrays to hold summary results
    Y_pred_test_all = []  # will be a nested list
    Y_pred_train_all = []  # will be a nested list
    Y_pred_test_labels_all = []
    Y_pred_train_labels_all = []
    train_err_arr = np.empty((len(train_set_durs), len(replicates)))
    test_err_arr = np.empty((len(train_set_durs), len(replicates)))
    train_lev_arr = np.empty((len(train_set_durs), len(replicates)))
    test_lev_arr = np.empty((len(train_set_durs), len(replicates)))
    train_syl_err_arr = np.empty((len(train_set_durs), len(replicates)))
    test_syl_err_arr = np.empty((len(train_set_durs), len(replicates)))

    NETWORKS = vak.network._load()

    for dur_ind, train_set_dur in enumerate(train_set_durs):

        Y_pred_test_this_dur = []
        Y_pred_train_this_dur = []
        Y_pred_test_labels_this_dur = []
        Y_pred_train_labels_this_dur = []

        for rep_ind, replicate in enumerate(replicates):
            print("getting train and test error for "
                  "training set with duration of {} seconds, "
                  "replicate {}".format(train_set_dur, replicate))

            training_records_dir = ('records_for_training_set_with_duration_of_'
                                    + str(train_set_dur) + '_sec_replicate_'
                                    + str(replicate))
            training_records_path = os.path.join(results_dirname,
                                                 training_records_dir)

            train_inds_file = glob(os.path.join(training_records_path, 'train_inds'))
            if len(train_inds_file) != 1:
                raise ValueError("incorrect number of train_inds files in {}, "
                                 "should only be one but found: {}"
                                 .format(training_records_path, train_inds_file))
            else:
                train_inds_file = train_inds_file[0]
            with open(os.path.join(train_inds_file), 'rb') as train_inds_file:
                train_inds = pickle.load(train_inds_file)

            # get training set
            Y_train_subset = Y_train[train_inds]
            X_train_subset = X_train[train_inds, :]
            assert Y_train_subset.shape[0] == X_train_subset.shape[0], \
                "mismatch between X and Y train subset shapes"
            if normalize_spectrograms:
                scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                               .format(train_set_dur, replicate))
                spect_scaler = joblib.load(
                    os.path.join(training_records_path, scaler_name))
                X_train_subset = spect_scaler.transform(X_train_subset)

            # Normalize before reshaping to avoid even more convoluted array reshaping.
            if normalize_spectrograms:
                X_test = spect_scaler.transform(X_test_copy)
            else:
                # get back "un-reshaped" X_test
                X_test = np.copy(X_test_copy)

            # need to get Y_test from copy because it gets reshaped every time through loop
            Y_test = np.copy(Y_test_copy)

            # save scaled spectrograms. Note we already saved training data scaled
            scaled_test_data_filename = os.path.join(summary_dirname,
                                                     'scaled_test_spects_duration_{}_replicate_{}'
                                                     .format(train_set_dur,
                                                             replicate))
            scaled_test_data_dict = {'X_test_scaled': X_test}
            joblib.dump(scaled_test_data_dict, scaled_test_data_filename)

            for net_name, net_config in zip(networks._fields, networks):
                # reload network #
                net_config_dict = net_config._asdict()
                net_config_dict['n_syllables'] = n_syllables
                net = NETWORKS[net_name](**net_config_dict)

                results_dirname_this_net = os.path.join(training_records_path, net_name)

                checkpoint_path = os.path.join(results_dirname_this_net, 'checkpoints')
                # we use latest checkpoint when doing summary for learncurve, assume that's "best trained"
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path)

                meta_file = glob(checkpoint_file + '*meta')
                if len(meta_file) != 1:
                    raise ValueError('Incorrect number of meta files for last saved checkpoint.\n'
                                     'For checkpoint {}, found these files:\n'
                                     '{}'
                                     .format(checkpoint_file, meta_file))
                else:
                    meta_file = meta_file[0]

                data_file = glob(checkpoint_file + '*data*')
                if len(data_file) != 1:
                    raise ValueError('Incorrect number of data files for last saved checkpoint.\n'
                                     'For checkpoint {}, found these files:\n'
                                     '{}'
                                     .format(checkpoint_file, data_file))
                else:
                    data_file = data_file[0]

                # reshape data for batching using net_config #
                (X_train_subset,
                 Y_train_subset,
                 num_batches_train) = utils.data.reshape_data_for_batching(
                    X_train_subset,
                    Y_train_subset,
                    net_config.batch_size,
                    net_config.time_bins)

                (X_test,
                 Y_test,
                 num_batches_test) = utils.data.reshape_data_for_batching(
                    X_test,
                    Y_test,
                    net_config.batch_size,
                    net_config.time_bins)

                scaled_reshaped_data_filename = os.path.join(summary_dirname,
                                                             'scaled_reshaped_spects_duration_{}_replicate_{}'
                                                             .format(train_set_dur,
                                                                     replicate))
                scaled_reshaped_data_dict = {
                    'X_train_subset_scaled_reshaped': X_train_subset,
                    'Y_train_subset_reshaped': Y_train_subset,
                    'X_test_scaled_reshaped': X_test,
                    'Y_test_reshaped': Y_test}
                joblib.dump(scaled_reshaped_data_dict,
                            scaled_reshaped_data_filename)

                with tf.Session(graph=net.graph) as sess:
                    tf.logging.set_verbosity(tf.logging.ERROR)

                    net.restore(sess=sess,
                                meta_file=meta_file,
                                data_file=data_file)

                    if 'Y_pred_train' in locals():
                        del Y_pred_train

                    print('calculating training set error')
                    for b in range(num_batches_train):  # "b" is "batch number"
                        d = {net.X: X_train_subset[:,
                                    b * net_config.time_bins: (b + 1) * net_config.time_bins, :],
                             net.lng: [net_config.time_bins] * net_config.batch_size}

                        if 'Y_pred_train' in locals():
                            preds = sess.run(net.predict, feed_dict=d)
                            preds = preds.reshape(net_config.batch_size, -1)
                            Y_pred_train = np.concatenate((Y_pred_train, preds),
                                                          axis=1)
                        else:
                            Y_pred_train = sess.run(net.predict, feed_dict=d)
                            Y_pred_train = Y_pred_train.reshape(net_config.batch_size, -1)

                    Y_train_subset = Y_train[train_inds]  # get back "unreshaped" Y_train_subset
                    # get rid of predictions to zero padding that don't matter
                    Y_pred_train = Y_pred_train.ravel()[:Y_train_subset.shape[0],
                                   np.newaxis]
                    train_err = np.sum(Y_pred_train - Y_train_subset != 0) / \
                                Y_train_subset.shape[0]
                    train_err_arr[dur_ind, rep_ind] = train_err
                    print('train error was {}'.format(train_err))
                    Y_pred_train_this_dur.append(Y_pred_train)

                    Y_train_subset_labels = utils.data.convert_timebins_to_labels(Y_train_subset,
                                                                                  labels_mapping)
                    Y_pred_train_labels = utils.data.convert_timebins_to_labels(Y_pred_train,
                                                                                labels_mapping)
                    Y_pred_train_labels_this_dur.append(Y_pred_train_labels)

                    if all([type(el) is int for el in Y_train_subset_labels]):
                        # if labels are ints instead of str
                        # convert to str just to calculate Levenshtein distance
                        # and syllable error rate.
                        # Let them be weird characters (e.g. '\t') because that doesn't matter
                        # for calculating Levenshtein distance / syl err rate
                        Y_train_subset_labels = ''.join(
                            [chr(el) for el in Y_train_subset_labels])
                        Y_pred_train_labels = ''.join(
                            [chr(el) for el in Y_pred_train_labels])

                    train_lev = metrics.levenshtein(Y_pred_train_labels,
                                                    Y_train_subset_labels)
                    train_lev_arr[dur_ind, rep_ind] = train_lev
                    print('Levenshtein distance for train set was {}'.format(
                        train_lev))
                    train_syl_err_rate = metrics.syllable_error_rate(
                        Y_train_subset_labels,
                        Y_pred_train_labels)
                    train_syl_err_arr[dur_ind, rep_ind] = train_syl_err_rate
                    print('Syllable error rate for train set was {}'.format(
                        train_syl_err_rate))

                    if 'Y_pred_test' in locals():
                        del Y_pred_test

                    print('calculating test set error')
                    for b in range(num_batches_test):  # "b" is "batch number"
                        d = {
                            net.X: X_test[:, b * net_config.time_bins: (b + 1) * net_config.time_bins, :],
                            net.lng: [net_config.time_bins] * net_config.batch_size}

                        if 'Y_pred_test' in locals():
                            preds = sess.run(net.predict, feed_dict=d)
                            preds = preds.reshape(net_config.batch_size, -1)
                            Y_pred_test = np.concatenate((Y_pred_test, preds),
                                                         axis=1)
                        else:
                            Y_pred_test = sess.run(net.predict, feed_dict=d)
                            Y_pred_test = Y_pred_test.reshape(net_config.batch_size, -1)

                    # again get rid of zero padding predictions
                    Y_pred_test = Y_pred_test.ravel()[:Y_test_copy.shape[0],
                                  np.newaxis]
                    test_err = np.sum(Y_pred_test - Y_test_copy != 0) / \
                               Y_test_copy.shape[0]
                    test_err_arr[dur_ind, rep_ind] = test_err
                    print('test error was {}'.format(test_err))
                    Y_pred_test_this_dur.append(Y_pred_test)

                    Y_pred_test_labels = utils.data.convert_timebins_to_labels(Y_pred_test,
                                                                               labels_mapping)
                    Y_pred_test_labels_this_dur.append(Y_pred_test_labels)
                    if all([type(el) is int for el in Y_pred_test_labels]):
                        # if labels are ints instead of str
                        # convert to str just to calculate Levenshtein distance
                        # and syllable error rate.
                        # Let them be weird characters (e.g. '\t') because that doesn't matter
                        # for calculating Levenshtein distance / syl err rate
                        Y_pred_test_labels = ''.join(
                            [chr(el) for el in Y_pred_test_labels])
                        # already converted actual Y_test_labels from int to str above,
                        # stored in variable `Y_test_labels_for_lev`

                    test_lev = metrics.levenshtein(Y_pred_test_labels,
                                                   Y_test_labels_for_lev)
                    test_lev_arr[dur_ind, rep_ind] = test_lev
                    print(
                        'Levenshtein distance for test set was {}'.format(test_lev))
                    test_syl_err_rate = metrics.syllable_error_rate(
                        Y_test_labels_for_lev,
                        Y_pred_test_labels)
                    print('Syllable error rate for test set was {}'.format(
                        test_syl_err_rate))
                    test_syl_err_arr[dur_ind, rep_ind] = test_syl_err_rate

        Y_pred_train_all.append(Y_pred_train_this_dur)
        Y_pred_test_all.append(Y_pred_test_this_dur)
        Y_pred_train_labels_all.append(Y_pred_train_labels_this_dur)
        Y_pred_test_labels_all.append(Y_pred_test_labels_this_dur)

    Y_pred_train_filename = os.path.join(summary_dirname,
                                         'Y_pred_train_all')
    with open(Y_pred_train_filename, 'wb') as Y_pred_train_file:
        pickle.dump(Y_pred_train_all, Y_pred_train_file)

    Y_pred_test_filename = os.path.join(summary_dirname,
                                        'Y_pred_test_all')
    with open(Y_pred_test_filename, 'wb') as Y_pred_test_file:
        pickle.dump(Y_pred_test_all, Y_pred_test_file)

    train_err_filename = os.path.join(summary_dirname,
                                      'train_err')
    with open(train_err_filename, 'wb') as train_err_file:
        pickle.dump(train_err_arr, train_err_file)

    test_err_filename = os.path.join(summary_dirname,
                                     'test_err')
    with open(test_err_filename, 'wb') as test_err_file:
        pickle.dump(test_err_arr, test_err_file)

    pred_and_err_dict = {'Y_pred_train_all': Y_pred_train_all,
                         'Y_pred_test_all': Y_pred_test_all,
                         'Y_pred_train_labels_all': Y_pred_train_labels_all,
                         'Y_pred_test_labels_all': Y_pred_test_labels_all,
                         'Y_train_labels': Y_train_labels,
                         'Y_test_labels': Y_test_labels,
                         'train_err': train_err_arr,
                         'test_err': test_err_arr,
                         'train_lev': train_lev_arr,
                         'train_syl_err_rate': train_syl_err_arr,
                         'test_lev': test_lev_arr,
                         'test_syl_err_rate': test_syl_err_arr,
                         'train_set_durs': train_set_durs,
                         'num_replicates': num_replicates}

    pred_err_dict_filename = os.path.join(summary_dirname,
                                          'y_preds_and_err_for_train_and_test')
    joblib.dump(pred_and_err_dict, pred_err_dict_filename)


if __name__ == '__main__':
    config_file = sys.argv[1]
    summary(config_file)
