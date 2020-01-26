import numpy as np

from ..util.path import array_dict_from_path
from ..util.validation import column_or_1d

from . import functional as F

__all__ = [
    'StandardizeSpect',
]


# adapted from:
# https://github.com/NickleDave/hybrid-vocal-classifier/blob/master/hvc/neuralnet/utils.py
class StandardizeSpect:
    """transform that standardizes spectrograms so they are all
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
        """initialize a new StandardizeSpect instance

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

    @classmethod
    def fit_df(cls, df, spect_key='s'):
        """fits StandardizeSpect instance, given a pandas DataFrame representing a dataset

        Parameters
        ----------
        df : pandas.DataFrame
            loaded from a .csv file representing a dataset, created by vak.io.dataframe.from_files
        spect_key : str
            key in files in 'spect_path' column that maps to spectrograms in arrays.
            Default is 's'.

        Returns
        -------
        standardize_spect : StandardizeSpect
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

    @classmethod
    def fit(cls, spect):
        """fit a StandardizeSpect instance.

        Parameters
        ----------
        spect : numpy.ndarray
            with dimensions (frequency bins, time bins)

        Notes
        -----
        Input should be spectrogram.
        Fit function finds the mean and standard deviation of each frequency bin,
        which are used by `transform` method to scale other spectrograms.
        """
        # TODO: make this function accept list and/or ndarray with batch dimension
        if spect.ndim != 2:
            raise ValueError('input spectrogram should be a 2-d array')

        mean_freqs = np.mean(spect, axis=1)
        std_freqs = np.std(spect, axis=1)
        assert mean_freqs.shape[-1] == spect.shape[0]
        assert std_freqs.shape[-1] == spect.shape[0]
        non_zero_std = np.argwhere(std_freqs != 0)
        return cls(mean_freqs, std_freqs, non_zero_std)

    def __call__(self, spect):
        """normalizes input spectrogram with fit parameters.

        Parameters
        ----------
        spect : numpy.ndarray
            2-d array with dimensions (frequency bins, time bins).

        Returns
        -------
        z_norm_spect : numpy.ndarray
            array standardized to same scale as set of spectrograms that
            SpectScaler was fit with
        """
        if any([not hasattr(self, attr) for attr in ['mean_freqs',
                                                     'std_freqs']]):
            raise AttributeError('SpectScaler properties are set to None,'
                                 'must call fit method first to set the'
                                 'value of these properties before calling'
                                 'transform')

        if type(spect) != np.ndarray:
            raise TypeError(
                f'type of spect must be numpy.ndarray but was: {type(spect)}'
            )

        if spect.shape[0] != self.mean_freqs.shape[0]:
            raise ValueError(f'number of rows in spects, {spect.shape[0]}, '
                             f'does not match number of elements in self.mean_freqs, {self.mean_freqs.shape[0]},'
                             'i.e. the number of frequency bins from the spectrogram'
                             'to which the scaler was fit originally')

        return F.standardize_spect(spect, self.mean_freqs, self.std_freqs, self.non_zero_std)
