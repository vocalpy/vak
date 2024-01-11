"""functions for making spectrogram

filters adapted from SciPy cookbook
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
spectrogram adapted from code by Kyle Kastner and Tim Sainburg
https://github.com/timsainb/python_spectrograms_and_inversion
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from matplotlib.mlab import specgram
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def spectrogram(
    dat: npt.NDArray,
    samp_freq: int,
    fft_size: int = 512,
    step_size: int = 64,
    thresh: float | None = None,
    transform_type: str | None = None,
    freq_cutoffs: list[int, int] | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    normalize: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """creates a spectrogram

    Parameters
    ----------
    dat : numpy.ndarray
        audio signal
    samp_freq : int
        sampling frequency in Hz
    fft_size : int
        size of window for Fast Fourier transform, number of time bins.
    step_size : int
        step size for Fast Fourier transform
    transform_type : str
        one of {'log_spect', 'log_spect_plus_one'}.
        'log_spect' transforms the spectrogram to log(spectrogram), and
        'log_spect_plus_one' does the same thing but adds one to each element.
        Default is None. If None, no transform is applied.
    thresh: int
        threshold minimum power for log spectrogram
    freq_cutoffs : tuple
        of two elements, lower and higher frequencies.
    min_val : float, optional
        Minimum value to allow in spectrogram.
        All values less than this will be set to this value.
        This operation is applied *after* the transform
        specified by ``transform_type``.
        Default is None.
    max_val : float, optional
        Maximum value to allow in spectrogram.
        All values greater than this will be set to this value.
        This operation is applied *after* the transform
        specified by ``transform_type``.
        Default is None.
    normalize : bool
        If True, min-max normalize the spectrogram.
        Normalization is done *after* the transform
        specified by ``transform_type``, and *after*
        the ``min_val`` and ``max_val`` operations.
        Default is False.

    Return
    ------
    spect : numpy.ndarray
        spectrogram
    freqbins : numpy.ndarray
        vector of centers of frequency bins from spectrogram
    timebins : numpy.ndarray
        vector of centers of time bins from spectrogram
    """
    noverlap = fft_size - step_size

    if freq_cutoffs:
        dat = butter_bandpass_filter(
            dat, freq_cutoffs[0], freq_cutoffs[1], samp_freq
        )

    # below only take [:3] from return of specgram because we don't need the image
    spect, freqbins, timebins = specgram(
        dat, fft_size, samp_freq, noverlap=noverlap
    )[:3]

    if transform_type:
        if transform_type == "log":
            spect = np.log(np.abs(spect) + np.finfo(spect).eps)
        elif transform_type == "log_spect":
            spect /= spect.max()  # volume normalize to max 1
            spect = np.log10(spect)  # take log
            if thresh:
                # I know this is weird, maintaining 'legacy' behavior
                spect[spect < -thresh] = -thresh
        elif transform_type == "log_spect_plus_one":
            spect = np.log10(spect + 1)
            if thresh:
                spect[spect < thresh] = thresh
    else:
        if thresh:
            spect[
                spect < thresh
            ] = thresh  # set anything less than the threshold as the threshold

    if min_val:
        spect[spect < min_val] = min_val
    if max_val:
        spect[spect > max_val] = max_val

    if normalize:
        s_max, s_min = spect.max(), spect.min()
        spect = (spect - s_min) / (s_max - s_min)
        spect = np.clip(spect, 0.0, 1.0)

    if freq_cutoffs:
        f_inds = np.nonzero(
            (freqbins >= freq_cutoffs[0]) & (freqbins < freq_cutoffs[1])
        )[
            0
        ]  # returns tuple
        spect = spect[f_inds, :]
        freqbins = freqbins[f_inds]

    return spect, freqbins, timebins
