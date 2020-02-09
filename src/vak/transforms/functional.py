import numpy as np


def standardize_spect(spect, mean_freqs, std_freqs, non_zero_std):
    """standardize spectrogram by subtracting off mean and dividing by standard deviation.

    Parameters
    ----------
    spect : numpy.ndarray
        with shape (frequencies, time bins)
    mean_freqs : numpy.ndarray
        vector of mean values for each frequency bin across the fit set of spectrograms
    std_freqs : numpy.ndarray
        vector of standard deviations for each frequency bin across the fit set of spectrograms
    non_zero_std : numpy.ndarray
        boolean, indicates where std_freqs has non-zero values. Used to avoid divide-by-zero errors.

    Returns
    -------
    transformed : numpy.ndarray
        with same shape as spect but with (approximately) zero mean and unit standard deviation
        (mean and standard devation will still vary by batch).
    """
    tfm = spect - mean_freqs[:, np.newaxis]  # need axis for broadcasting
    # keep any stds that are zero from causing NaNs
    tfm[non_zero_std, :] = tfm[non_zero_std, :] / std_freqs[non_zero_std, np.newaxis]
    return tfm
