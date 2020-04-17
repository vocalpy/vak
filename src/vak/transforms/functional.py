import numpy as np
import torch

__all__ = [
    'pad_to_window',
    'standardize_spect',
    'to_floattensor',
    'to_longtensor',
    'view_as_window_batch'
]


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


def pad_to_window(arr, window_size, padval=0., return_padding_mask=True):
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
    if arr.ndim == 1:
        width = arr.shape[0]
    elif arr.ndim == 2:
        height, width = arr.shape
    else:
        raise ValueError(
            f'input array must be 1d or 2d but number of dimensions was: {arr.ndim}'
        )

    target_width = int(
        np.ceil(width / window_size) * window_size
    )

    if arr.ndim == 1:
        padded = np.ones((target_width,)) * padval
        padded[:width] = arr
    elif arr.ndim == 2:
        padded = np.ones((height, target_width)) * padval
        padded[:, :width] = arr

    if return_padding_mask:
        padding_mask = np.zeros((target_width,), dtype=np.bool)
        padding_mask[:width] = True
        return padded, padding_mask
    else:
        return padded


def view_as_window_batch(arr, window_width):
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
        with shape (batch size, window_size) if array is 1d,
        or with shape (batch size, height, window_size) if array is 2d.
        Batch size will be arr.shape[-1] // window_width.
        Window width must divide arr.shape[-1] evenly.
        To pad the array so it can be divided into windows of the specified
        width, use the `pad_to_window` transform

    Notes
    -----
    adapted from skimage.util.view_as_blocks
    https://github.com/scikit-image/scikit-image/blob/f1b7cf60fb80822849129cb76269b75b8ef18db1/skimage/util/shape.py#L9
    """
    if not(type(window_width) == int and window_width > 0):
        raise ValueError(
            f'window width must be a positive integer'
        )

    if arr.ndim == 1:
        window_shape = (window_width,)
    elif arr.ndim == 2:
        height, _ = arr.shape
        window_shape = (height, window_width)
    else:
        raise ValueError(
            f'input array must be 1d or 2d but number of dimensions was: {arr.ndim}'
        )

    window_shape = np.array(window_shape)
    arr_shape = np.array(arr.shape)
    if (arr_shape % window_shape).sum() != 0:
        raise ValueError("'window_width' does not divide evenly into with 'arr' shape. "
                         "Use 'pad_to_window' transform to pad array so it can be windowed.")

    new_shape = tuple(arr_shape // window_shape) + tuple(window_shape)
    new_strides = tuple(arr.strides * window_shape) + arr.strides
    batch_windows = np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)
    # when 2d, first dim 1 because new shape has height equal to original arr
    if batch_windows.ndim == 4 and batch_windows.shape[0] == 1:
        # we don't want that extra dim of size 1, we want first dim to be "batch"
        batch_windows = np.squeeze(batch_windows)
    return batch_windows


def to_floattensor(arr):
    """convert Numpy array to torch.FloatTensor.

    Parameters
    ----------
    arr : numpy.ndarray

    Returns
    -------
    float_tensor
        with dtype 'float32'
    """
    return torch.from_numpy(arr).float()


def to_longtensor(arr):
    """convert Numpy array to torch.LongTensor.

    Parameters
    ----------
    arr : numpy.ndarray

    Returns
    -------
    long_tensor : torch.Tensor
        with dtype 'float64'
    """
    return torch.from_numpy(arr).long()


def add_channel(input, channel_dim=0):
    """add a channel dimension to a tensor.
    Transform that makes it easy to treat a spectrogram as an image,
    by adding a dimension with a single 'channel', analogous to grayscale.
    In this way the tensor can be fed to e.g. convolutional layers.

    Parameters
    ----------
    input : torch.Tensor
    channel_dim : int
        dimension where "channel" is added. Default is 0.
    """
    return torch.unsqueeze(input, dim=channel_dim)
