import numpy as np

import vak.timebins


def test_timebin_dur_from_vecs():
    timebin_dur = 0.001
    time_bins = np.linspace(0.0, 5.0, num=int(5 / timebin_dur))
    computed = vak.timebins.timebin_dur_from_vec(
        time_bins=time_bins, n_decimals_trunc=3
    )
    assert timebin_dur == computed
