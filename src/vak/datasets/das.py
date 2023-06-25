"""Datasets used to benchmark Deep Audio Segmenter.

References
----------
https://data.goettingen-research-online.de/dataverse/das
"""
import numpy as np


def get_das_default_data_stride(kernel_size: int, nb_conv: int, num_samples: int) -> int:
    """Compute default stride that ``das.data.AudioSequence`` uses for audio windows,
    given the kernel size and downsampling factor ``nb_conv``.

    Parameters
    ----------
    kernel_size
    nb_conv
    num_samples

    Returns
    -------

    Notes
    -----
    This function specifies a stride smaller than the input size to the network,
    to mitigate edge effects that might occur if the network only saw
    non-overlapping consecutive windows during training.
    See https://github.com/janclemenslab/das/blob/3f3bf76b705e0960d5fd84f26033a4fa3cde2472/src/das/train.py#L232
    """
    data_padding = np.ceil(kernel_size * nb_conv)
    return num_samples - 2 * data_padding
