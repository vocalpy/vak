from glob import glob
import os

import crowsetta
import joblib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..dataset import VocalizationDataset
from .. import network
from ..utils.data import reshape_data_for_batching
from ..utils.labels import lbl_tb2labels


def predict(predict_vds_path,
            checkpoint_path,
            networks,
            labelmap,
            spect_scaler_path=None,
            save_predict_vds=True
            ):
    """make predictions with one trained model

    Parameters
    ----------
    predict_vds_path : str, list
        path or list of paths to VocalizationDataset(s) for which
        annotation should be predicted.
    checkpoint_path : str
        path to directory with saved model
    networks : dict
        where each key is the name of a neural network and the corresponding
        value is the configuration for that network (in a namedtuple or a dict)
    labelmap : dict
        maps set of labels for vocalizations to consecutive integer
        values {0,1,2,...N}, where N is the number of classes, i.e., label types
    spect_scaler_path : str
        path to a saved SpectScaler object used to normalize spectrograms.
        If spectrograms were normalized and this is not provided, will give
        incorrect results.
        Default is None.
    save_predict_vds : bool
        if True, save VocalizationDataset after predicting labels.
        Default is True.

    Returns
    -------
    None
    """
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError('directory {}, specified as '
                                'checkpoint_path, is not found.'
                                .format(checkpoint_path))

    if spect_scaler_path:
        if not os.path.isfile(spect_scaler_path):
            raise FileNotFoundError(
                f'file for spect_scaler not found at path:\n{spect_scaler_path}'
            )
        spect_scaler = joblib.load(spect_scaler_path)
    else:
        spect_scaler = None

    if type(predict_vds_path) is str:
        predict_vds_path = [predict_vds_path]
    elif type(predict_vds_path) is list:
        pass
    else:
        raise TypeError(
            'predict_vds_path should be a string path or list '
            f'of string paths, but type was {type(predict_vds_path)}'
        )

    n_classes = len(labelmap)  # used below when instantiating network
    predict_vds_out = []
    for p_vds_path in predict_vds_path:
        predict_vds = VocalizationDataset.load(json_fname=p_vds_path)

        if predict_vds.are_spects_loaded() is False:
            predict_vds = predict_vds.load_spects()

        X_data = predict_vds.spects_list()
        X_data_spect_ID_vector = np.concatenate(
            [np.ones((spect.shape[-1],), dtype=np.int64) * ind
             for ind, spect in enumerate(X_data)]
        )
        X_data = np.concatenate(X_data, axis=1)

        timebin_dur = set([voc.metaspect.timebin_dur for voc in predict_vds.voc_list])
        if len(timebin_dur) > 1:
            raise ValueError(
                'found more than one time bin duration in '
                f'VocalizationDataset: {timebin_dur}'
            )
        elif len(timebin_dur) == 1:
            timebin_dur = timebin_dur.pop()
        else:
            raise ValueError(
                f'invalid time bin durations from training set: {timebin_dur}'
            )

        # transpose X_data, so rows are timebins and columns are frequency bins
        # because networks expect this orientation for input
        X_data = X_data.T
        freq_bins = X_data.shape[-1]  # needed for net config

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
                             .format(checkpoint_path))
        elif len(data_file) < 1:
            raise ValueError('did not find .data file in {}'
                             .format(checkpoint_path))
        else:
            data_file = data_file[0]

        NETWORKS = network._load()

        for net_name, net_config in networks.items():
            net_config_dict = net_config._asdict()
            net_config_dict['n_syllables'] = n_classes
            if 'freq_bins' in net_config_dict:
                net_config_dict['freq_bins'] = freq_bins
            net = NETWORKS[net_name](**net_config_dict)

            if spect_scaler:
                X_data = spect_scaler.transform(X_data)

            (Xd_batch,
             num_batches) = reshape_data_for_batching(X_data,
                                                      batch_size=net_config.batch_size,
                                                      time_steps=net_config.time_bins)

            time_bins = net_config.time_bins  # for brevity below
            with tf.Session(graph=net.graph) as sess:
                tf.logging.set_verbosity(tf.logging.ERROR)

                net.restore(sess=sess,
                            meta_file=meta_file,
                            data_file=data_file)

                Y_pred = []
                pbar = tqdm(range(num_batches))
                for b in pbar:
                    pbar.set_description(
                        f'Predicting labels for {predict_vds_path}'
                    )

                    d = {net.X: Xd_batch[:, b * time_bins: (b + 1) * time_bins, :],
                         net.lng: [time_bins] * net_config.batch_size}

                    # if Y_pred exists, we concatenate with new predictions
                    # for next batch
                    preds = sess.run(net.predict, feed_dict=d)
                    preds = preds.reshape(-1)  # batch_size
                    Y_pred.append(preds)

                # remove zero padding added by reshape_data_for_batching function
                Y_pred = np.concatenate(Y_pred)
                Y_pred = Y_pred[0:X_data.shape[0]]

            lbl_tbs = [Y_pred[X_data_spect_ID_vector == ind]
                       for ind in np.unique(X_data_spect_ID_vector)]
            for lbl_tb, voc in zip(lbl_tbs, predict_vds.voc_list):
                voc.metaspect.lbl_tb = lbl_tb
                # TODO: change this when Crowsetta uses Annotation instead of Sequence
                # annot_dict = voc.annot.as_dict()
                labels = lbl_tb2labels(lbl_tb, labelmap)
                voc.annot = crowsetta.Sequence.from_dict(annot_dict)

            if save_predict_vds:
                predict_vds.save(json_fname=p_vds_path)

            predict_vds_out.append(predict_vds)

    return predict_vds_out
