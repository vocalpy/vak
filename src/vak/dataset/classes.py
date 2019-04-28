import os

import numpy as np
import attr
from attr.validators import optional, instance_of

from ..utils import spect


@attr.s
class Spectrogram:
    """class to represent a spectrogram"""
    array = attr.ib(validator=instance_of(np.ndarray))
    audio_file = attr.ib(validator=[instance_of(str), os.path.isfile])
    freq_bins = attr.ib(validator=instance_of(np.ndarray))
    time_bins = attr.ib(validator=instance_of(np.ndarray))


@attr.s
class Vocalization:
    """class to represent an annotated vocalization

    Attributes
    ----------
    spect : numpy.ndarray
        spectrogram, in a numpy array
    spect_file : str
        path to file containing spectrogram as an array
    audio_file : str
        path to file containing audio
    annotation : list, tuple, or crowsetta.Sequence
        annotations of vocalizations for files
    """
    spect = attr.ib(validator=optional(instance_of(Spectrogram)), default=None)
    spect_file = attr.ib(validator=optional(instance_of(str)), default=None)
    audio_file = attr.ib(validator=instance_of(str), default=None)
    annotations = list()


@attr.s
class VocalSet:
    """class to represent a dataset of annotated vocalizations"""
    set = attr.ib()
    @set.validator
    def all_voc(self, attribute, value):
        if not all([type(element) == Vocalization for element in value]):
            raise TypeError(f'all ')

    @classmethod
    def from_audio(cls, audio_format, spect_params, annot_format, dir=None, files=None):
        """create a dataset of vocalizations from audio files and annotations.
        In the process, create spectrograms from the audio files as well.

        Parameters
        ----------
        audio_format
        annot_format
        dir
        files

        Returns
        -------

        """
        if dir and files:
            raise ValueError('must specify either dir or files, not both')

        if dir:
            files = list_from_dir(dir, audio_format)

        spect_list = spect.from_list(files)

        for spect,

        return cls()

    @classmethod
        def from_audio(cls, audio_format, spect_params, annot_format, dir=None, files=None):
