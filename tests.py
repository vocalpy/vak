# run at the command line by typing `python -m unittest`

import unittest

import numpy as np

from cnn_bilstm.utils import reshape_data_for_batching


class TestUtils(unittest.TestCase):

    def test_reshape_data_for_batching(self):
        """test that method for batching data does not change Y_pred in some way
        """
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
         num_batches) = reshape_data_for_batching(X_in, Y_in, batch_size,
                                                  time_steps, input_vec_size)

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
