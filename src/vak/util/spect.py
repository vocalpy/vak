"""spectrogram utilities
filters adapted from SciPy cookbook
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
spectrogram adapted from code by Kyle Kastner and Tim Sainburg
https://github.com/timsainb/python_spectrograms_and_inversion
"""
import numpy as np

from scipy.signal import butter, lfilter
from matplotlib.mlab import specgram

from ..io.spect import array_dict_from_path

from .validation import column_or_1d


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
    """class that transforms spectrograms so they are all 
    on the same scale, by subtracting off the mean and dividing by the 
    standard deviation from a 'fit' set of spectrograms.
    
    Attributes
    ----------
    mean_freqs : numpy.ndarray
        mean values for each frequency bin across the fit set of spectrograms
    std_freqs : numpy.ndarray
        standard deciation for each frequency bin across the fit set of spectrograms
    non_zero_std : numpy.ndarray
        boolean, indicates where std_freqs has non-zero values. Used to avoid divide-by-zero errors.
    """
    def __init__(self,
                 mean_freqs=None,
                 std_freqs=None,
                 non_zero_std=None):
        """initialize a new SpectScaler instance

        Parameters
        ----------
        mean_freqs : numpy.ndarray
            vector of mean values for each frequency bin across the fit set of spectrograms
        std_freqs : numpy.ndarray
            vector of standard deviations for each frequency bin across the fit set of spectrograms
        non_zero_std : numpy.ndarray
            boolean, indicates where std_freqs has non-zero values. Used to avoid divide-by-zero errors.
        """
        if any([arg is not None for arg in (mean_freqs, std_freqs, non_zero_std)]):
            mean_freqs, std_freqs, non_zero_std = (
                column_or_1d(arr) for arr in (mean_freqs, std_freqs, non_zero_std)
            )
            if len(np.unique([arg.shape[0] for arg in (mean_freqs, std_freqs, non_zero_std)])) != 1:
                raise ValueError(
                    f'mean_freqs, std_freqs, and non_zero_std must all have the same length'
                )

        self.mean_freqs = mean_freqs
        self.std_freqs = std_freqs
        self.non_zero_std = non_zero_std

    def fit(self, spect):
        """fit a SpectScaler.

        Parameters
        ----------
        spect : numpy.ndarray
            with dimensions (time bins, frequency bins)

        Notes
        -----
        Input should be spectrogram, oriented so that the columns are frequency bins.
        Fit function finds the mean and standard deviation of each frequency bin,
        which are used by `transform` method to scale other spectrograms.
        """
        if spect.ndim != 2:
            raise ValueError('input spectrogram should be a 2-d array')

        self.mean_freqs = np.mean(spect, axis=0)
        self.std_freqs = np.std(spect, axis=0)
        assert self.mean_freqs.shape[-1] == spect.shape[-1]
        assert self.std_freqs.shape[-1] == spect.shape[-1]
        self.non_zero_std = np.argwhere(self.std_freqs != 0)

    def _transform(self, spect):
        """helper function that transforms input spectrogram by subtracting off fit mean
        and then dividing by standard deviation

        Parameters
        ----------
        spect : numpy.ndarray

        Returns
        -------
        transformed : numpy.ndarray
        """
        tfm = spect - self.mean_freqs[:, np.newaxis]  # need axis for broadcasting
        # keep any stds that are zero from causing NaNs
        return tfm[self.non_zero_std, :] / self.std_freqs[self.non_zero_std, np.newaxis]

    def transform(self, spects):
        """normalizes input spectrograms with fit parameters.

        Parameters
        ----------
        spects : numpy.ndarray, list
            2-d array or list of 2-d arrays with dimensions (frequency bins. time bins).

        Returns
        -------
        z_norm_spects : numpy.ndarray, list
            array or list of arrays, standardized to same scale as set of spectrograms that
            SpectScaler was fit with

        Notes
        -----
        assumes spectrograms are oriented with columns being frequency bins
        and rows being time bins.
        """
        if any([not hasattr(self, attr) for attr in ['mean_freqs',
                                                     'std_freqs']]):
            raise AttributeError('SpectScaler properties are set to None,'
                                 'must call fit method first to set the'
                                 'value of these properties before calling'
                                 'transform')

        if type(spects) != np.ndarray and type(spects) != list:
            raise TypeError(
                f'type {type(spects)} is not valid for spects'
            )

        if type(spects) == np.ndarray:
            if spects.shape[0] != self.mean_freqs.shape[0]:
                raise ValueError(f'number of rows in spects, {spects.shape[0]}, '
                                 f'does not match number of elements in self.mean_freqs, {self.mean_freqs.shape[0]},'
                                 'i.e. the number of frequency bins from the spectrogram'
                                 'to which the scaler was fit originally')
            return self._transform(spects)

        elif type(spects) == list:
            z_norm_spects = []
            for spect in spects:
                z_norm_spects.append(self._transform(spect))

            return z_norm_spects

    def fit_transform(self, spects):
        """first calls fit and then returns normalized spects
        transformed using the fit parameters

        Parameters
        ----------
        spects : numpy.ndarray or list
            2-d numpy array or list of 2-d numpy arrays with dimensions (time bins, frequency bins).

        Returns
        -------
        z_norm_spects : numpy.ndarray or list
            array or list of arrays, with mean subtracted off each frequency bin and then divided by
            standard deviation
        """
        if type(spects) != np.ndarray:
            raise TypeError('spects passed to fit_transform '
                            'should be numpy array, not {}'
                            .format(type(spects)))

        if spects.ndim != 2:
            raise ValueError('ndims of spects should be 2, not {}'
                             .format(spects.ndim))

        self.fit(spects)
        return self.transform(spects)

    @classmethod
    def fit_df(cls, df, spect_key='s'):
        """fits a SpectScaler, given a pandas DataFrame representing a dataset

        Parameters
        ----------
        df : pandas.DataFrame
        spect_key : str
            key in files in 'spect_path' column that maps to spectrograms in arrays.
            Default is 's'.

        Returns
        -------
        SpectScaler
            instance fit to spectrograms in df
        """
        spect_paths = df['spect_path']
        spect = array_dict_from_path(spect_paths[0])[spect_key]
        # in files, spectrograms are in orientation (freq bins, time bins)
        # so we take mean and std across columns, i.e. time bins, i.e. axis 1
        mean_freqs = np.mean(spect, axis=1)
        std_freqs = np.std(spect, axis=1)

        for spect_path in spect_paths[1:]:
            spect = array_dict_from_path(spect_path)[spect_key]
            mean_freqs += np.mean(spect, axis=1)
            std_freqs += np.std(spect, axis=1)
        mean_freqs = mean_freqs / len(spect_paths)
        std_freqs = std_freqs / len(spect_paths)
        non_zero_std = np.argwhere(std_freqs != 0)
        return cls(mean_freqs, std_freqs, non_zero_std)
