import os
import sys
from configparser import ConfigParser
from glob import glob

import joblib
import numpy as np
import tensorflow as tf

from vak.utils.data import reshape_data_for_batching


def predict(results_dirname,
            dir_to_predict,
            checkpoint_dir,
            ):
    """make predictions with one trained model

    Parameters
    ----------
    results_dirname
    dir_to_predict
    checkpoint_dir

    Returns
    -------

    """
    if not os.path.isdir(dir_to_predict):
        raise FileNotFoundError('directory {}, specified as '
                                'dir_to_predict, is not found.'
                                .format(dir_to_predict))
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError('directory {}, specified as '
                                'checkpoint_dir, is not found.'
                                .format(checkpoint_dir))
    meta_file = glob(os.path.join(checkpoint_dir,
                                  'checkpoint*meta*'))
    if len(meta_file) > 1:
        raise ValueError('found more than one .meta file in {}'
                         .format(checkpoint_dir))
    elif len(meta_file) < 1:
        raise ValueError('did not find .meta file in {}'
                         .format(checkpoint_dir))
    else:
        meta_file = meta_file[0]

    data_file = glob(os.path.join(checkpoint_dir,
                                  'checkpoint*data*'))
    if len(data_file) > 1:
        raise ValueError('found more than one .data file in {}'
                         .format(checkpoint_dir))
    elif len(data_file) < 1:
        raise ValueError('did not find .data file in {}'
                         .format(checkpoint_dir))
    else:
        data_file = data_file[0]

    # TODO should be able to just call dataset here, right?
    # instead of forcing user to specify spect_file_list
    # should give them option to do either
    spect_file_list = glob(os.path.join(dir_to_predict,
                                        '*.spect'))
    if spect_file_list == []:
        raise ValueError('did not find any .spect files in {}'
                         .format(dir_to_predict))

    model = TweetyNet(n_syllables=n_syllables,
        input_vec_size=input_vec_size,
        batch_size=batch_size)

    with tf.Session(graph=model.graph) as sess:
        tf.logging.set_verbosity(tf.logging.ERROR)

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
