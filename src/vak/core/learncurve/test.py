import os
import logging
import pickle
from datetime import datetime
from glob import glob

import joblib
import numpy as np
import tensorflow as tf

from ... import metrics, utils
from ... import network
from ...dataset import VocalizationDataset


def test(results_dirname,
         test_vds_path,
         train_vds_path,
         networks,
         train_set_durs,
         num_replicates,
         output_dir=None,
         normalize_spectrograms=False,
         save_transformed_data=False):
    """generate summary learning curve from networks trained by cli.learncurve
    Computes error on test set for each network trained by learncurve,
    and saves in a ./summary directory within results_dir

    Parameters
    ----------
    results_dirname : str
        path to directory containing results created by a run of learncurve.train
    test_vds_path : str
        path to VocalizationDataset that represents test data
    train_vds_path : str
        path to VocalizationDataset that represents training data
    networks : dict
        where each key is the name of a neural network and the corresponding
        value is the configuration for that network (in a namedtuple or a dict)
    train_set_durs : list
        of int, durations in seconds of subsets taken from training data
        to create a learning curve, e.g. [5, 10, 15, 20]
    num_replicates : int
        number of times to replicate training for each training set duration
        to better estimate mean accuracy for a training set of that size.
        Each replicate uses a different randomly drawn subset of the training
        data (but of the same duration).
    output_dir : str
        path to directory where output should be saved. Default is None, in which
        case results are saved in results_dirname.
    normalize_spectrograms : bool
        if True, use spect.utils.data.SpectScaler to normalize the spectrograms.
        Normalization is done by subtracting off the mean for each frequency bin
        of the training set and then dividing by the std for that frequency bin.
        This same normalization is then applied to validation + test data.
    save_transformed_data : bool
        if True, save transformed data (i.e. scaled, reshaped).
        Useful if you need to check what the data looks like when fed to networks.

    Returns
    -------
    None
    """
    # ---------------- pre-conditions ----------------------------------------------------------------------------------
    if not os.path.isdir(results_dirname):
        raise NotADirectoryError(
            f'directory specified as results_dirname not found: {results_dirname}'
        )

    if output_dir:
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(
                f'specified output directory not found: {output_dir}'
            )

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    if output_dir:
        test_dirname = os.path.join(output_dir, 'test')
    else:
        test_dirname = os.path.join(results_dirname, 'test')
    os.makedirs(test_dirname)

    # ---------------- logging -----------------------------------------------------------------------------------------
    logger = logging.getLogger('learncurve.test')

    if logging.getLevelName(logger.level) != 'INFO':
        logger.setLevel('INFO')

    logger.info(f"Logging run of learncurve.test to '{test_dirname}'")

    # ---------------- load training data  -----------------------------------------------------------------------------
    logger.info('Loading training VocalizationDataset from {}'.format(
        os.path.dirname(
            train_vds_path)))
    train_vds = VocalizationDataset.load(json_fname=train_vds_path)

    if train_vds.are_spects_loaded() is False:
        train_vds = train_vds.load_spects()

    X_train = train_vds.spects_list()
    X_train = np.concatenate(X_train, axis=1)
    Y_train = train_vds.lbl_tb_list()
    Y_train = np.concatenate(Y_train)
    # transpose so rows are time bins
    X_train = X_train.T

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

    # ---------------- load test data  -----------------------------------------------------------------------------
    logger.info('Loading test VocalizationDataset from {}'.format(
        os.path.dirname(
            test_vds_path)))
    test_vds = VocalizationDataset.load(json_fname=test_vds_path)

    if test_vds.are_spects_loaded() is False:
        test_vds = test_vds.load_spects()

    if test_vds.labelmap != train_vds.labelmap:
        raise ValueError(
            f'labelmap of test set, {test_vds.labelmap}, does not match labelmap of training set, '
            f'{train_vds.labelmap}'
        )

    def unpack_test():
        """helper function because we want to get back test set unmodified every time we go through
        main loop below, without copying giant arrays"""
        X_test = test_vds.spects_list()
        X_test = np.concatenate(X_test, axis=1)
        # transpose so rows are time bins
        X_test = X_test.T
        Y_test = test_vds.lbl_tb_list()
        Y_test = np.concatenate(Y_test)
        return X_test, Y_test

    # just get X_test to make sure it has the right shape
    X_test, _ = unpack_test()
    if X_train.shape[-1] != X_test.shape[-1]:
        raise ValueError(f'Number of frequency bins in training set spectrograms, {X_train.shape[-1]}, '
                         f'does not equal number in test set spectrograms, {X_test.shape[-1]}.')
    freq_bins = X_test.shape[-1]  # number of columns

    # concatenate labels into one big string
    # used for Levenshtein distance + syllable error rate
    Y_train_labels = [voc.annot.labels.tolist() for voc in train_vds.voc_list]
    Y_train_labels_for_lev = ''.join([chr(lbl) if type(lbl) is int else lbl
                                      for labels in Y_train_labels for lbl in labels])
    Y_test_labels = [voc.annot.labels.tolist() for voc in test_vds.voc_list]
    Y_test_labels_for_lev = ''.join([chr(lbl) if type(lbl) is int else lbl
                                     for labels in Y_train_labels for lbl in labels])

    replicates = range(1, num_replicates + 1)
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

    NETWORKS = network._load()

    for dur_ind, train_set_dur in enumerate(train_set_durs):

        Y_pred_test_this_dur = []
        Y_pred_train_this_dur = []
        Y_pred_test_labels_this_dur = []
        Y_pred_train_labels_this_dur = []

        for rep_ind, replicate in enumerate(replicates):
            logger.info(
                "getting train and test error for training set with duration of "
                f"{train_set_dur} seconds, replicate {replicate}"
            )

            training_records_dir = ('records_for_training_set_with_duration_of_'
                                    + str(train_set_dur) + '_sec_replicate_'
                                    + str(replicate))
            training_records_path = os.path.join(results_dirname,
                                                 'train',
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

            X_test, Y_test = unpack_test()
            # Normalize before reshaping to avoid even more convoluted array reshaping.
            if normalize_spectrograms:
                X_test = spect_scaler.transform(X_test)

            if save_transformed_data:
                scaled_test_data_filename = os.path.join(test_dirname,
                                                         'scaled_test_spects_duration_{}_replicate_{}'
                                                         .format(train_set_dur,
                                                                 replicate))
                scaled_test_data_dict = {'X_test_scaled': X_test}
                joblib.dump(scaled_test_data_dict, scaled_test_data_filename)

            for net_name, net_config in networks.items():
                # reload network #
                net_config_dict = net_config._asdict()
                net_config_dict['n_syllables'] = n_classes
                if 'freq_bins' in net_config_dict:
                    net_config_dict['freq_bins'] = freq_bins
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
                # Notice we don't reshape Y_train
                (X_train_subset,
                 _,
                 num_batches_train) = utils.data.reshape_data_for_batching(
                    X_train_subset,
                    net_config.batch_size,
                    net_config.time_bins,
                    Y_train_subset)

                # Notice we don't reshape Y_test
                (X_test,
                 _,
                 num_batches_test) = utils.data.reshape_data_for_batching(
                    X_test,
                    net_config.batch_size,
                    net_config.time_bins,
                    Y_test)

                if save_transformed_data:
                    scaled_reshaped_data_filename = os.path.join(test_dirname,
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

                    logger.info('calculating training set error')
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

                    # get rid of predictions to zero padding that don't matter
                    Y_pred_train = Y_pred_train.ravel()[:Y_train_subset.shape[0], np.newaxis]
                    train_err = np.sum(Y_pred_train != Y_train_subset) / Y_train_subset.shape[0]
                    train_err_arr[dur_ind, rep_ind] = train_err
                    logger.info('train error was {}'.format(train_err))
                    Y_pred_train_this_dur.append(Y_pred_train)

                    Y_train_subset_labels = utils.data.convert_timebins_to_labels(Y_train_subset,
                                                                                  train_vds.labelmap)
                    Y_pred_train_labels = utils.data.convert_timebins_to_labels(Y_pred_train,
                                                                                train_vds.labelmap)
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
                    logger.info('Levenshtein distance for train set was {}'.format(
                        train_lev))
                    train_syl_err_rate = metrics.syllable_error_rate(Y_train_subset_labels,
                                                                     Y_pred_train_labels)
                    train_syl_err_arr[dur_ind, rep_ind] = train_syl_err_rate
                    logger.info('Syllable error rate for train set was {}'.format(
                        train_syl_err_rate))

                    if 'Y_pred_test' in locals():
                        del Y_pred_test

                    logger.info('calculating test set error')
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
                    Y_pred_test = Y_pred_test.ravel()[:Y_test.shape[0], np.newaxis]
                    test_err = np.sum(Y_pred_test != Y_test) / Y_test.shape[0]
                    test_err_arr[dur_ind, rep_ind] = test_err
                    logger.info('test error was {}'.format(test_err))
                    Y_pred_test_this_dur.append(Y_pred_test)

                    Y_pred_test_labels = utils.data.convert_timebins_to_labels(Y_pred_test,
                                                                               test_vds.labelmap)
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
                    logger.info(
                        'Levenshtein distance for test set was {}'.format(test_lev))
                    test_syl_err_rate = metrics.syllable_error_rate(
                        Y_test_labels_for_lev,
                        Y_pred_test_labels)
                    logger.info('Syllable error rate for test set was {}'.format(
                        test_syl_err_rate))
                    test_syl_err_arr[dur_ind, rep_ind] = test_syl_err_rate

        Y_pred_train_all.append(Y_pred_train_this_dur)
        Y_pred_test_all.append(Y_pred_test_this_dur)
        Y_pred_train_labels_all.append(Y_pred_train_labels_this_dur)
        Y_pred_test_labels_all.append(Y_pred_test_labels_this_dur)

    Y_pred_train_filename = os.path.join(test_dirname,
                                         'Y_pred_train_all')
    with open(Y_pred_train_filename, 'wb') as Y_pred_train_file:
        pickle.dump(Y_pred_train_all, Y_pred_train_file)

    Y_pred_test_filename = os.path.join(test_dirname,
                                        'Y_pred_test_all')
    with open(Y_pred_test_filename, 'wb') as Y_pred_test_file:
        pickle.dump(Y_pred_test_all, Y_pred_test_file)

    train_err_filename = os.path.join(test_dirname,
                                      'train_err')
    with open(train_err_filename, 'wb') as train_err_file:
        pickle.dump(train_err_arr, train_err_file)

    test_err_filename = os.path.join(test_dirname,
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

    pred_err_dict_filename = os.path.join(test_dirname,
                                          'y_preds_and_err_for_train_and_test')
    joblib.dump(pred_and_err_dict, pred_err_dict_filename)
