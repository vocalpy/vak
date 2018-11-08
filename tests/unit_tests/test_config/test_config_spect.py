"""tests for songdeck.config.spectrogram module"""
import unittest
from configparser import ConfigParser

import songdeck.config.spectrogram
import songdeck.utils


def _base_config():
    base_config = ConfigParser()
    base_config['SPECTROGRAM'] = {
        'fft_size': '512',
        'step_size': '64',
        'freq_cutoffs': '500, 10000',
        'thresh': '6.25',
        'transform_type': 'log_spect',
    }
    return base_config


class TestParseSpectConfig(unittest.TestCase):

    def setUp(self):
        self.get_config = _base_config()

    def test_config_tuple_has_all_attrs(self):
        config_obj = self.get_config
        spect_config_tup = songdeck.config.spectrogram.parse_spect_config(config_obj)
        for field in songdeck.config.spectrogram.SpectConfig._fields:
            self.assertTrue(hasattr(spect_config_tup, field))

    def test_thresh_default(self):
        # test that thresh option is added
        # if we don't specify it, and set to None
        config_obj = ConfigParser()
        config_obj['SPECTROGRAM'] = {
            'fft_size': '512',
            'step_size': '64',
            'freq_cutoffs': '500, 10000',
        }
        spect_config_tup = songdeck.config.spectrogram.parse_spect_config(config_obj)
        self.assertTrue(spect_config_tup.thresh is None)


if __name__ == '__main__':
    unittest.main()
