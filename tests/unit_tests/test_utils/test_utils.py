import unittest

import vak.utils.utils
from vak.utils.utils import OnlyValDurError, InvalidDurationError, SplitsDurationGreaterThanDatasetDurationError


class TestUtils(unittest.TestCase):
    def test_validate_durs(self):
        train_dur_in = 100
        val_dur_in = 25
        test_dur_in = 75
        vds_dur = 200
        train_dur_out, val_dur_out, test_dur_out = vak.utils.split._validate_durs(train_dur_in,
                                                                                  val_dur_in,
                                                                                  test_dur_in,
                                                                                  vds_dur)
        self.assertTrue(
            all([out_ == in_ for out_, in_ in zip((train_dur_in, val_dur_in, test_dur_in),
                                                  (train_dur_out, val_dur_out, test_dur_out))]))

        train_dur_in = 100
        val_dur_in = None
        test_dur_in = None
        vds_dur = 200
        train_dur_out, val_dur_out, test_dur_out = vak.utils.split._validate_durs(train_dur_in,
                                                                                  val_dur_in,
                                                                                  test_dur_in,
                                                                                  vds_dur)
        self.assertTrue(
            all([train_dur_out == train_dur_in,
                 test_dur_out == -1,
                 val_dur_out is None]))

        train_dur_in = None
        val_dur_in = None
        test_dur_in = 100
        vds_dur = 200
        train_dur_out, val_dur_out, test_dur_out = vak.utils.split._validate_durs(train_dur_in,
                                                                                  val_dur_in,
                                                                                  test_dur_in,
                                                                                  vds_dur)
        self.assertTrue(
            all([train_dur_out == -1,
                 test_dur_out == test_dur_in,
                 val_dur_out is None]))

        train_dur_in = None
        val_dur_in = None
        test_dur_in = None
        vds_dur = 200
        with self.assertRaises(ValueError):
            # because we have to specify at least one of train_dur or test_dur
            vak.utils.split._validate_durs(train_dur_in,
                                           val_dur_in,
                                           test_dur_in,
                                           vds_dur)

        train_dur_in = None
        val_dur_in = 100
        test_dur_in = None
        vds_dur = 200
        # because we only specified duration for validation set
        with self.assertRaises(OnlyValDurError):
            vak.utils.split._validate_durs(train_dur_in, val_dur_in, test_dur_in, vds_dur)

        train_dur_in = -2
        test_dur_in = None
        val_dur_in = 100
        vds_dur = 200
        # because negative duration is invalid
        with self.assertRaises(InvalidDurationError):
            vak.utils.split._validate_durs(train_dur_in, val_dur_in, test_dur_in, vds_dur)

        train_dur_in = 100
        test_dur_in = 100
        val_dur_in = 100
        vds_dur = 200
        # because total splits duration is greater than dataset duration
        with self.assertRaises(SplitsDurationGreaterThanDatasetDurationError):
            vak.utils.split._validate_durs(train_dur_in, val_dur_in, test_dur_in, vds_dur)


if __name__ == '__main__':
    unittest.main()
