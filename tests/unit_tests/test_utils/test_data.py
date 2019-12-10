from pathlib import Path
import unittest

import numpy as np

from vak.io import Dataset
import vak.utils.data


HERE = Path(__file__).parent
TEST_DATA_DIR = HERE.joinpath('..', '..', 'test_data')


class TestData(unittest.TestCase):

    def setUp(self):
        test_data_vds_path = TEST_DATA_DIR.joinpath('vds')
        self.test_data_spects_path = test_data_vds_path.glob('spectrograms_*')
        self.test_data_spects_path = list(self.test_data_spects_path)[0]
        self.train_vds_path = test_data_vds_path.glob('*prep*train.vds.json')
        self.train_vds_path = list(self.train_vds_path)[0]

    def test_get_inds_for_dur(self):
        train_vds = Dataset.load(json_fname=self.train_vds_path)
        train_vds = train_vds.load_spects()

        X_train = train_vds.spects_list()
        spect_ID_vec = np.concatenate(
            [np.ones((spect.shape[-1],), dtype=np.int64) * ind
             for ind, spect in enumerate(X_train)]
        )
        Y_train = train_vds.lbl_tb_list()
        lt = np.concatenate(Y_train)
        lm = train_vds.labelmap

        target_dur = 6  # seconds
        timebin_dur = set([voc.metaspect.timebin_dur
                           for voc in train_vds.voc_list])
        timebin_dur = timebin_dur.pop()

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
