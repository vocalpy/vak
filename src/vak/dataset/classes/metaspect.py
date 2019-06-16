import numpy as np
import attr
from attr.validators import instance_of, optional

from ...utils.general import timebin_dur_from_vec
from .validators import asarray_if_not


@attr.s(cmp=False)
class MetaSpect:
    """class to represent a spectrogram and 'metadata' associated with it,
    such as the vectors of frequency and time bin centers, and things that
    are more vocalization specific, like a vector of labels for each time bin.

    Will typically correspond to a single file, e.g. a .mat or .npz file that
    contains the spectrogram and associated arrays.

    Attributes
    ----------
    spect : numpy.ndarray
        spectrogram contained in an array
    freq_bins : numpy.ndarray
        vector of frequencies in spectrogram, where each value is a bin center.
    time_bins : numpy.ndarray
        vector of times in spectrogram, where each value is a bin center.
    timebin_dur : numpy.ndarray
        duration of a timebin in seconds from spectrogram
    lbl_tb : numpy.ndarray
        labeled time bins, i.e. result of taking labels, onsets and offsets of
        segments from some annotation file and then converting them into a vector
        using the `vak.utils.labels.label_timebin` function
    """
    spect = attr.ib(validator=instance_of(np.ndarray), converter=asarray_if_not)
    freq_bins = attr.ib(validator=instance_of(np.ndarray), converter=asarray_if_not)
    time_bins = attr.ib(validator=instance_of(np.ndarray), converter=asarray_if_not)
    timebin_dur = attr.ib(validator=optional(instance_of(float)), default=None)
    lbl_tb = attr.ib(validator=optional(instance_of(np.ndarray)), converter=asarray_if_not, default=None)

    @classmethod
    def from_dict(cls,
                  spect_file_dict,
                  spect_key='s',
                  freqbins_key='f',
                  timebins_key='t',
                  timebin_dur=None,
                  n_decimals_trunc=3):
        """create a Spectrogram instance from a dictionary-like object that
        provides access to arrays loaded from a file, e.g. a .mat or .npz file

        Parameters
        ----------
        spect_file_dict : dict-like
            dictionary-like object providing access to .mat or .npz file that contains
            a spectrogram and associated arrays
        freqbins_key : str
            key for accessing vector of frequency bins in files. Default is 'f'.
        timebins_key : str
            key for accessing vector of time bins in files. Default is 't'.
        spect_key : str
            key for accessing spectrogram in files. Default is 's'.
        timebin_dur : float
            duration of time bins. Default is None. If None, then
        n_decimals_trunc : int
            number of decimal places to keep when truncating the timebin duration calculated from
            the spectrogram arrays.
            Default is 3, i.e. assumes milliseconds is the last significant digit.

        Returns
        -------
        spect : vak.dataset.classes.MetaSpect
            a Spectrogram instance with attributes freq_bins, time_bins, array, and timebin_dur
        """
        if timebin_dur is None:
            timebin_dur = timebin_dur_from_vec(time_bins=spect_file_dict[timebins_key],
                                               n_decimals_trunc=n_decimals_trunc)

        return cls(freq_bins=spect_file_dict[freqbins_key],
                   time_bins=spect_file_dict[timebins_key],
                   spect=spect_file_dict[spect_key],
                   timebin_dur=timebin_dur)
