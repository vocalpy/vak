import os
from glob import glob
import unittest

import numpy as np
import joblib

import vak.utils.data


HERE = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(HERE, '..', '..', 'test_data')


class TestData(unittest.TestCase):

    def setUp(self):
        self.test_data_spects_path = glob(os.path.join(TEST_DATA_DIR,
                                                  'spects',
                                                  'spectrograms_*'))[0]
        self.train_data_path = os.path.join(self.test_data_spects_path, 'train_data_dict')

    def test_get_inds_for_dur(self):
        tdd = joblib.load(self.train_data_path)
        spect_ID_vec = tdd['spect_ID_vector']
        lt = tdd['Y_train']
        lm = tdd['labels_mapping']
        target_dur = 6  # seconds
        timebin_dur = tdd['timebin_dur']
        inds_to_use = vak.utils.data.get_inds_for_dur(spect_ID_vector=spect_ID_vec,
                                                      labeled_timebins_vector=lt,
                                                      labels_mapping=lm,
                                                      target_duration=target_dur,
                                                      timebin_dur_in_s=timebin_dur,
                                                      max_iter=1000,
                                                      method='incfreq')
        self.assertTrue(timebin_dur * inds_to_use.shape[0] == target_dur)

    def test_reshape_data_for_batching(self):
        """test that method for batching data does not change Y_pred in some way"""
        input_vec_size = 513  # number of frequency bins in spectrogram
        rows = 5000
        X_in = np.empty((rows, input_vec_size))  # doesn't matter what's in X
        Y_in = np.random.randint(low=0, high=10, size=(rows, 1))
        # below constants because that's what's used in code
        batch_size = 11
        time_steps = 87
        # split up so that we can loop through `num_batches` of batches
        # where each batch has one dimension of size `time_steps`
        (X_out,
         Y_out,
         num_batches) = vak.utils.data.reshape_data_for_batching(X_in,
                                                                 batch_size,
                                                                 time_steps,
                                                                 Y_in)

        self.assertEqual(Y_out.shape[0], batch_size)

        # get `predictions` that are just reshaping of the data
        for b in range(num_batches):
            Y_chunk = Y_out[:, b * time_steps: (b + 1) * time_steps]
            if 'Y_pred_concat' in locals():
                # in real code we pass Y_chunk to
                # our network to get predicted values
                # Here we just get values back the way we do from the network
                Y_pred = Y_chunk.ravel()
                Y_pred = Y_pred.reshape(batch_size, -1)
                Y_pred_concat = np.concatenate((Y_pred_concat, Y_pred), axis=1)
            else:
                Y_pred = Y_chunk.ravel()
                Y_pred = Y_pred.reshape(batch_size, -1)
                Y_pred_concat = Y_pred

        # did `Y_in` survive dis-assembly then re-assembly?
        # if so, then low accuracy from models
        # can't be due to some weird matrix reshaping bug
        Y_pred_concat = Y_pred_concat.ravel()
        Y_pred_concat = Y_pred_concat[:Y_in.shape[0], np.newaxis]

        self.assertTrue(np.array_equal(Y_pred_concat, Y_in))


if __name__ == '__main__':
    unittest.main()
