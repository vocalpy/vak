import os
from glob import glob
from datetime import datetime
import pickle

import joblib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..utils.data import reshape_data_for_batching, make_spects_from_list_of_files
from ..utils.mat import convert_mat_to_spect
from .. import network


def predict(checkpoint_path,
            networks,
            labels_mapping_path,
            spect_params,
            dir_to_predict=None,
            mat_spect_files_path=None,
            spect_scaler_path=None
            ):
    """make predictions with one trained model

    Parameters
    ----------
    checkpoint_path : str
        path to directory with saved model
    networks : dict
        where each key is the name of a neural network and the corresponding
        value is the configuration for that network (in a namedtuple or a dict)
    labels_mapping_path : str
        path to file that contains labels mapping, to convert output from consecutive
        digits back to labels used for audio segments (e.g. birdsong syllables)
    spect_params : dict
        Dictionary of parameters for creating spectrograms.
    dir_to_predict : str
        path to directory where input files are located
    mat_spect_files_path
        path to directory with .mat files containing spectrograms that should be used
        as data for which predictions are made
    spect_scaler_path : str
        path to a saved SpectScaler object used to normalize spectrograms.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.
        Default is None.

    Returns
    -------
    None
    """
    timenow = datetime.now().strftime('%y%m%d_%H%M%S')

    if dir_to_predict is None and mat_spect_files_path is None:
        raise ValueError('must specify either dir_to_predict or mat_spect_files_path but both are None')

    if dir_to_predict and mat_spect_files_path:
        raise ValueError('got values for both dir_to_predict and mat_spect_files_path, not clear which to use')

    with open(labels_mapping_path, 'rb') as labels_map_file_obj:
        labels_mapping = pickle.load(labels_map_file_obj)
    n_syllables = len(labels_mapping)

    if mat_spect_files_path:
        print('will use spectrograms from .mat files in {}'
              .format(mat_spect_files_path))
        mat_spect_files = glob(os.path.join(mat_spect_files_path, '*.mat'))
        spects_dir = os.path.join(mat_spect_files_path,
                                  'spectrograms_' + timenow)
        os.mkdir(spects_dir)
        spect_files_path = convert_mat_to_spect(mat_spect_files,
                                                mat_spects_annotation_file,
                                                spects_dir,
                                                labels_mapping=labels_mapping)
        dir_to_predict = spect_files_path

    else:
        if not os.path.isdir(dir_to_predict):
            raise FileNotFoundError('directory {}, specified as '
                                    'dir_to_predict, is not found.'
                                    .format(dir_to_predict))

        spects_dir = os.path.join(dir_to_predict,
                              'spectrograms_' + timenow)
        os.mkdir(spects_dir)

        cbins = glob(os.path.join(dir_to_predict, '*.cbin'))
        if cbins == []:
            # if we don't find .cbins in data_dir, look in sub-directories
            cbins = []
            subdirs = glob(os.path.join(dir_to_predict, '*/'))
            for subdir in subdirs:
                cbins.extend(glob(os.path.join(dir_to_predict,
                                               subdir,
                                               '*.cbin')))
        if cbins == []:
            # try looking for .wav files
            wavs = glob(os.path.join(dir_to_predict, '*.wav'))

            if cbins == [] and wavs == []:
                raise FileNotFoundError('No .cbin or .wav files found in {} or'
                                        'immediate sub-directories'
                                        .format(dir_to_predict))

        if cbins:
            spect_files_path = \
                make_spects_from_list_of_files(cbins,
                                               spect_params,
                                               spects_dir,
                                               labels_mapping,
                                               skip_files_with_labels_not_in_labelset=False,
                                               is_for_predict=True)
        elif wavs:
            spect_files_path = \
                make_spects_from_list_of_files(wavs,
                                               spect_params,
                                               spects_dir,
                                               labels_mapping,
                                               skip_files_with_labels_not_in_labelset=False,
                                               is_for_predict=True)

    # TODO should be able to just call dataset here, right?
    # instead of forcing user to specify spect_file_list
    # should give them option to do either
    spect_file_list = glob(os.path.join(spects_dir,
                                        '*.spect'))
    if spect_file_list == []:
        raise ValueError('did not find any .spect files in {}'
                         .format(dir_to_predict))

    X_data = []
    for spect_file in spect_file_list:
        spect_dict = joblib.load(spect_file)
        X_data.append(spect_dict['spect'].T)
    freq_bins_all_spects = [Xd.shape[-1] for Xd in X_data]
    uniq_freq_bins = set(freq_bins_all_spects)
    if len(uniq_freq_bins) != 1:
        raise ValueError('Found spectrograms with different numbers of frequency bins in dir_to_predict.\n'
                         f'Different values for numbers of frequency bins found were: {uniq_freq_bins}')
    else:
        freq_bins = list(uniq_freq_bins)[0]

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError('directory {}, specified as '
                                'checkpoint_path, is not found.'
                                .format(checkpoint_path))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    meta_file = glob(checkpoint_file + '*meta*')
    if len(meta_file) > 1:
        raise ValueError('found more than one .meta file in {}'
                         .format(checkpoint_path))
    elif len(meta_file) < 1:
        raise ValueError('did not find .meta file in {}'
                         .format(checkpoint_path))
    else:
        meta_file = meta_file[0]

    data_file = glob(checkpoint_file + '*data*')
    if len(data_file) > 1:
        raise ValueError('found more than one .data file in {}'
                         .format(checkpoint_dir))
    elif len(data_file) < 1:
        raise ValueError('did not find .data file in {}'
                         .format(checkpoint_dir))
    else:
        data_file = data_file[0]

    num_spect_files = len(spect_file_list)

    NETWORKS = network._load()

    for net_name, net_config in networks.items():
        net_config_dict = net_config._asdict()
        net_config_dict['n_syllables'] = n_syllables
        if 'freq_bins' in net_config_dict:
            net_config_dict['freq_bins'] = freq_bins
        net = NETWORKS[net_name](**net_config_dict)

        if spect_scaler_path:
            spect_scaler = joblib.load(spect_scaler_path)

        with tf.Session(graph=net.graph) as sess:
            tf.logging.set_verbosity(tf.logging.ERROR)

            net.restore(sess=sess,
                        meta_file=meta_file,
                        data_file=data_file)

            preds_dict = {}
            pbar = tqdm(spect_file_list)
            for file_num, spect_file in enumerate(pbar):
                pbar.set_description(
                    f'Predicting labels for {os.path.basename(spect_file)}, file {file_num} of {num_spect_files}'
                )

                Xd = X_data[file_num]
                if spect_scaler_path:
                    Xd = spect_scaler.transform(Xd)
                (Xd_batch,
                 num_batches) = reshape_data_for_batching(Xd,
                                                          batch_size=net_config.batch_size,
                                                          time_steps=net_config.time_bins)

                if 'Y_pred' in locals():
                    del Y_pred
                # work through current spectrogram batch by batch
                for b in range(num_batches):  # "b" is "batch number"
                    d = {net.X: Xd_batch[:, b * net_config.time_bins: (b + 1) * net_config.time_bins, :],
                         net.lng: [net_config.time_bins] * net_config.batch_size}
                    if 'Y_pred' in locals():
                        # if Y_pred exists, we concatenate with new predictions
                        # for next batch
                        preds = sess.run(net.predict, feed_dict=d)
                        preds = preds.reshape(-1)  # batch_size
                        Y_pred = np.concatenate((Y_pred, preds), axis=0)
                    else:  # if Y_pred doesn't exist yet
                        Y_pred = sess.run(net.predict, feed_dict=d)
                        Y_pred = Y_pred.reshape(-1)

                # remove zero padding added by reshape_data_for_batching function
                Y_pred = Y_pred.ravel()
                Y_pred = Y_pred[0:Xd.shape[0]]
                preds_dict[spect_file] = Y_pred

        fname = os.path.join(dir_to_predict, 'predictions')
        joblib.dump(preds_dict, fname)
