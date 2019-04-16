import os
from glob import glob
import unittest

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


if __name__ == '__main__':
    unittest.main()
