import os
import sys
from configparser import ConfigParser
from glob import glob

import joblib
import numpy as np
import tensorflow as tf

from cnn_bilstm.utils import reshape_data_for_batching
from cnn_bilstm import CNNBiLSTM


def predict(config_file):
    """predict segmentation and labels with one trained CNN-BiLSTM model"""
    if not config_file.endswith('.ini'):
        raise ValueError('{} is not a valid config file, must have .ini extension'
                         .format(config_file))
    config = ConfigParser()
    config.read(config_file)
    print('Using definitions in: ' + config_file)
    results_dirname = config['OUTPUT']['results_dir_made_by_main_script']


    dir_to_predict = config['PREDICT']['dir_to_predict']
    spect_file_list = glob(os.path.join(dir_to_predict,
                                        '*.spect'))
    if spect_file_list == []:
        raise ValueError('did not find any .spect files in {}'
                         .format(dir_to_predict))

    batch_size = int(config['NETWORK']['batch_size'])
    time_steps = int(config['NETWORK']['time_steps'])
    n_syllables = int(config['NETWORK']['n_syllables'])
    input_vec_size = int(config['NETWORK']['input_vec_size'])


    model = CNNBiLSTM(n_syllables=n_syllables,
        input_vec_size=input_vec_size,
        batch_size=batch_size)

    with tf.Session(graph=model.graph) as sess:
        tf.logging.set_verbosity(tf.logging.ERROR)

        checkpoint_dir = config['PREDICT']['checkpoint_dir']
        meta_file = glob(os.path.join(checkpoint_dir,
                                      'checkpoint*meta*'))
        if len(meta_file) > 1:
            raise ValueError('found more than one .meta file in {}'
                             .format(checkpoint_dir))
        else:
            meta_file = meta_file[0]

        data_file = glob(os.path.join(checkpoint_dir,
                                      'checkpoint*data*'))
        if len(data_file) > 1:
            raise ValueError('found more than one .data file in {}'
                             .format(checkpoint_dir))
        else:
            data_file = data_file[0]

        model.restore(sess=sess,
        meta_file=meta_file,
        data_file=data_file)

        num_spect_files = len(spect_file_list)
        preds_dict = {}
        for file_num, spect_file in enumerate(spect_file_list):
            print('Predicting labels for {}, file {} of {}'
                  .format(spect_file, file_num, num_spect_files))

            data = joblib.load(spect_file)
            Xd = data['spect'].T
            Yd = data['labeled_timebins']
            (Xd_batch,
             Yd_batch,
             num_batches) = reshape_data_for_batching(Xd,
                                                      Yd,
                                                      batch_size,
                                                      time_steps,
                                                      input_vec_size)

            if 'Y_pred' in locals():
                del Y_pred
            # work through current spectrogram batch by batch
            for b in range(num_batches):  # "b" is "batch number"
                d = {model.X:Xd_batch[:, b * time_steps: (b + 1) * time_steps, :],
                     model.lng: [time_steps] * batch_size}
                if 'Y_pred' in locals():
                    # if Y_pred exists, we concatenate with new predictions
                    # for next batch
                    preds = sess.run(model.predict, feed_dict=d)
                    preds = preds.reshape( -1)  # batch_size
                    Y_pred = np.concatenate((Y_pred, preds), axis=0)
                else:  # if Y_pred doesn't exist yet
                    Y_pred = sess.run(model.predict, feed_dict=d)
                    Y_pred = Y_pred.reshape(-1)

            # remove zero padding added by reshape_data_for_batching function
            Y_pred = Y_pred.ravel()
            Y_pred = Y_pred[0:len(Yd)]
            preds_dict[spect_file] = Y_pred

    fname = os.path.join(dir_to_predict, 'predictions')
    joblib.dump(preds_dict, fname)


if __name__ == '__main__':
    config_file = sys.argv[1]
    predict(config_file)
