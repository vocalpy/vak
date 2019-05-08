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


def spectrogram(dat, samp_freq, fft_size=512, step_size=64, thresh=None, transform_type=None,
                freq_cutoffs=None):
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
        dat = butter_bandpass_filter(dat,
                                     freq_cutoffs[0],
                                     freq_cutoffs[1],
                                     samp_freq)

    # below only take [:3] from return of specgram because we don't need the image
    spect, freqbins, timebins = specgram(dat, fft_size, samp_freq, noverlap=noverlap)[:3]

    if transform_type:
        if transform_type == 'log_spect':
            spect /= spect.max()  # volume normalize to max 1
            spect = np.log10(spect)  # take log
            if thresh:
                # I know this is weird, maintaining 'legacy' behavior
                spect[spect < -thresh] = -thresh
        elif transform_type == 'log_spect_plus_one':
            spect = np.log10(spect + 1)
            if thresh:
                spect[spect < thresh] = thresh
    else:
        if thresh:
            spect[spect < thresh] = thresh  # set anything less than the threshold as the threshold

    if freq_cutoffs:
        f_inds = np.nonzero((freqbins >= freq_cutoffs[0]) &
                            (freqbins < freq_cutoffs[1]))[0]  # returns tuple
        spect = spect[f_inds, :]
        freqbins = freqbins[f_inds]

    return spect, freqbins, timebins


# adapted from:
# https://github.com/NickleDave/hybrid-vocal-classifier/blob/master/hvc/neuralnet/utils.py
class SpectScaler:
    """class that scales spectrograms that all have the
    same number of frequency bins. Any input spectrogram
    will be scaled by subtracting off the mean of each
    frequency bin from the 'fit' set of spectrograms, and
    then dividing by the standard deviation of each
    frequency bin from the 'fit' set.
    """
    def __init__(self):
        self.columnMeans = None
        self.columnStds = None
        self.nonZeroStd = None

    def fit(self, spect):
        """fit a SpectScaler.
        Input should be spectrogram,
        oriented so that the columns are frequency bins.
        Fit function finds the mean and standard deviation of
        each frequency bin, which are used by `transform` method
        to scale other spectrograms.

        Parameters
        ----------
        spect : 2-d numpy array
            with dimensions (time bins, frequency bins)
        """
        if spect.ndim != 2:
            raise ValueError('input spectrogram should be a 2-d array')

        self.columnMeans = np.mean(spect, axis=0)
        self.columnStds = np.std(spect, axis=0)
        assert self.columnMeans.shape[-1] == spect.shape[-1]
        assert self.columnStds.shape[-1] == spect.shape[-1]
        self.nonZeroStd = np.argwhere(self.columnStds != 0)

    def _transform(self, spect):
        """transforms input spectrogram by subtracting off fit mean
        and then dividing by standard deviation
        """
        transformed = spect - self.columnMeans
        # to keep any zero stds from causing NaNs
        transformed[:, self.nonZeroStd] = (
            transformed[:, self.nonZeroStd] / self.columnStds[self.nonZeroStd])
        return transformed

    def transform(self, spects):
        """normalizes input spectrograms with fit parameters
        Assumes spectrograms are oriented with columns being frequency bins
        and rows being time bins.

        Parameters
        ----------
        spects : 2-d numpy array or list of 2-d numpy arrays
            with dimensions (time bins, frequency bins)

        """
        if any([not hasattr(self, attr) for attr in ['columnMeans',
                                                     'columnStds']]):
            raise AttributeError('SpectScaler properties are set to None,'
                                 'must call fit method first to set the'
                                 'value of these properties before calling'
                                 'transform')

        if type(spects) != np.ndarray and type(spects) != list:
            raise TypeError('type {} is not valid for spects'
                            .format(type(spects)))

        if type(spects) == np.ndarray:
            if spects.shape[-1] != self.columnMeans.shape[-1]:
                raise ValueError('number of columns in spects, {}, '
                                 'does not match shape of self.columnMeans, {},'
                                 'i.e. the number of columns from the spectrogram'
                                 'to which the scaler was fit originally')
            return self._transform(spects)

        elif type(spects) == list:
            z_norm_spects = []
            for spect in spects:
                z_norm_spects.append(self._transform(spect))

            return z_norm_spects

    def fit_transform(self, spects):
        """first calls fit and then returns normalized spects
        transformed using the fit parameters"""

        if type(spects) != np.ndarray:
            raise TypeError('spects passed to fit_transform '
                            'should be numpy array, not {}'
                            .format(type(spects)))

        if spects.ndim != 2:
            raise ValueError('ndims of spects should be 2, not {}'
                             .format(spects.ndim))

        self.fit(spects)
        return self.transform(spects)
