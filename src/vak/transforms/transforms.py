import numpy as np

from .. import files
from ..validators import column_or_1d

from . import functional as F

__all__ = [
    'AddChannel',
    'PadToWindow',
    'StandardizeSpect',
    'ToFloatTensor',
    'ToLongTensor',
    'ViewAsWindowBatch',
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
        standard deviation for each frequency bin across the fit set of spectrograms
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
        spect = files.spect.load(spect_paths[0])[spect_key]
        # in files, spectrograms are in orientation (freq bins, time bins)
        # so we take mean and std across columns, i.e. time bins, i.e. axis 1
        mean_freqs = np.mean(spect, axis=1)
        std_freqs = np.std(spect, axis=1)

        for spect_path in spect_paths[1:]:
            spect = files.spect.load(spect_path)[spect_key]
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

    def __repr__(self):
        args = f'(mean_freqs={self.mean_freqs}, std_freqs={self.std_freqs}, non_zero_std={self.non_zero_std})'
        return self.__class__.__name__ + args


class PadToWindow:
    """pad a 1d or 2d array so that it can be reshaped
    into consecutive windows of specified size

    Parameters
    ----------
    arr : numpy.ndarray
        with 1 or 2 dimensions, e.g. a vector of labeled timebins
        or a spectrogram.
    window_size : int
        width of window in number of elements.
    padval : float
        value to pad with. Added to end of array, the
        "right side" if 2-dimensional.
    return_padding_mask : bool
        if True, return a boolean vector to use for cropping
        back down to size before padding. padding_mask has size
        equal to width of padded array, i.e. original size
        plus padding at the end, and has values of 1 where
        columns in padded are from the original array,
        and values of 0 where columns were added for padding.

    Returns
    -------
    padded : numpy.ndarray
        padded with padval
    padding_mask : np.bool
        has size equal to width of padded, i.e. original size
        plus padding at the end. Has values of 1 where
        columns in padded are from the original array,
        and values of 0 where columns were added for padding.
        Only returned if return_padding_mask is True.
    """
    def __init__(self, window_size, padval=0., return_padding_mask=True):
        if not (type(window_size) == int) or (type(window_size) == float and window_size.is_integer() is False):
            raise ValueError(
                f'window size must be an int or a whole number float;'
                f' type was {type(window_size)} and value was {window_size}'
            )

        if type(padval) not in (int, float):
            raise TypeError(
                f'type for padval must be int or float but was: {type(padval)}'
            )
        if not type(return_padding_mask) == bool:
            raise TypeError(
                'return_padding_mask must be boolean (True or False), '
                f'but was type {type(return_padding_mask)} with value {return_padding_mask}'
            )

        self.window_size = window_size
        self.padval = padval
        self.return_padding_mask = return_padding_mask

    def __call__(self, arr):
        return F.pad_to_window(arr, self.window_size, self.padval, self.return_padding_mask)

    def __repr__(self):
        args = f'(window_size={self.window_size}, padval={self.padval}, return_padding_mask={self.return_padding_mask})'
        return self.__class__.__name__ + args


class ViewAsWindowBatch:
    """return view of a 1d or 2d array as a batch of non-overlapping windows

    Parameters
    ----------
    arr : numpy.ndarray
        with 1 or 2 dimensions, e.g. a vector of labeled timebins
        or a 2-d array representing a spectrogram.
        If the array has 2-d dimensions, the returned array will
        have dimensions (batch, height of array, window width)
    window_width : int
        width of window in number of elements.

    Returns
    -------
    batch_windows : numpy.ndarray
        with shape (batch size, window_width) if array is 1d,
        or with shape (batch size, height, window_width) if array is 2d.
        Batch size will be arr.shape[-1] // window_width.
        Window width must divide arr.shape[-1] evenly.
        To pad the array so it can be divided into windows of the specified
        width, use the `pad_to_window` transform

    Notes
    -----
    adapted from skimage.util.view_as_blocks
    https://github.com/scikit-image/scikit-image/blob/f1b7cf60fb80822849129cb76269b75b8ef18db1/skimage/util/shape.py#L9
    """
    def __init__(self, window_width):
        if not (type(window_width) == int) or (type(window_width) == float and window_width.is_integer() is False):
            raise ValueError(
                f'window size must be an int or a whole number float;'
                f' type was {type(window_width)} and value was {window_width}'
            )

        self.window_width = window_width

    def __call__(self, arr):
        return F.view_as_window_batch(arr, self.window_width)

    def __repr__(self):
        args = f'(window_width={self.window_width})'
        return self.__class__.__name__ + args


class ToFloatTensor:
    """convert Numpy array to torch.FloatTensor.

    Parameters
    ----------
    arr : numpy.ndarray

    Returns
    -------
    float_tensor
        with dtype 'float32'
    """
    def __init__(self):
        pass

    def __call__(self, arr):
        return F.to_floattensor(arr)

    def __repr__(self):
        return self.__class__.__name__


class ToLongTensor:
    """convert Numpy array to torch.LongTensor.

    Parameters
    ----------
    arr : numpy.ndarray

    Returns
    -------
    long_tensor : torch.Tensor
        with dtype 'float64'
    """
    def __init__(self):
        pass

    def __call__(self, arr):
        return F.to_longtensor(arr)

    def __repr__(self):
        return self.__class__.__name__


class AddChannel:
    """add a channel dimension to a 2-dimensional tensor.
    Transform that makes it easy to treat a spectrogram as an image,
    by adding a dimension with a single 'channel', analogous to grayscale.
    In this way the tensor can be fed to e.g. convolutional layers.

    Parameters
    ----------
    input : torch.Tensor
        with two dimensions (height, width).
    channel_dim : int
        dimension where "channel" is added.
        Default is 0, which returns a tensor with dimensions (channel, height, width).
    """
    def __init__(self, channel_dim=0):
        if not (type(channel_dim) == int) or (type(channel_dim) == float and channel_dim.is_integer() is False):
            raise ValueError(
                f'window size must be an int or a whole number float;'
                f' type was {type(channel_dim)} and value was {channel_dim}'
            )

        channel_dim = int(channel_dim)

        if channel_dim < 0 and channel_dim != -1:
            raise ValueError(
                'value of channel_dim should be a non-negative integer, or -1 (for last dimension). '
                f'Value was: {channel_dim}'
            )

        self.channel_dim = channel_dim

    def __call__(self, input):
        return F.add_channel(input, channel_dim=self.channel_dim)

    def __repr__(self):
        args = f'(channel_dim={self.channel_dim})'
        return self.__class__.__name__ + args
