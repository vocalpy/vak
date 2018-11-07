"""spectrogram utilities
filters adapted from SciPy cookbook
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
spectrogram adapted from code by Kyle Kastner and Tim Sainburg
https://github.com/timsainb/python_spectrograms_and_inversion
"""

import numpy as np
from scipy.signal import butter, lfilter
from matplotlib.mlab import specgram


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def spectrogram(data, samp_freq, fft_size=512, step_size=64, thresh=None, transform_type=None):
    """creates a spectrogram

    Parameters
    ----------
    data : ndarray
        audio signal
    log_transform: bool
        if True, take the log of the spectrogram
    thresh: int
        threshold minimum power for log spectrogram

    Return
    ------
    spec : ndarray

    freqbins : ndarray

    timebins : ndarray
    """

    noverlap = fft_size - step_size

    # below only take [:3] from return of specgram because we don't need the image
    spec, freqbins, timebins = specgram(data, fft_size, samp_freq, noverlap=noverlap)[:3]

    if transform_type:
        if transform_type == 'log_spect':
            spec /= spec.max()  # volume normalize to max 1
            spec = np.log10(spec)  # take log
            if thresh:
                # I know this is weird, maintaining 'legacy' behavior
                spec[spec < -thresh] = -thresh
        elif transform_type == 'log_spect_plus_one':
            spec = np.log10(spec + 1)
            if thresh:
                spec[spec < thresh] = thresh
    else:
        if thresh:
            spec[spec < thresh] = thresh  # set anything less than the threshold as the threshold

    return spec, freqbins, timebins
