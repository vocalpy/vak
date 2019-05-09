import json

import numpy as np
import attr
from attr.validators import optional, instance_of
from crowsetta import Sequence


@attr.s(cmp=False)
class Spectrogram:
    """class to represent a spectrogram

    Attributes
    ----------
    freq_bins : numpy.ndarray
        vector of frequencies in spectrogram, where each value is a bin center.
    time_bins : numpy.ndarray
        vector of times in spectrogram, where each value is a bin center.
    timebin_dur : numpy.ndarray
        duration of a timebin in seconds from spectrogram
    duration : numpy.ndarray
        duration of spectrogram, i.e. len(time_bins) * timebin_dur
    array : numpy.ndarray
        spectrogram contained in an array
    """
    freq_bins = attr.ib(validator=instance_of(np.ndarray))
    time_bins = attr.ib(validator=instance_of(np.ndarray))
    timebin_dur = attr.ib(validator=instance_of(float))
    duration = attr.ib(validator=instance_of(float))
    array = attr.ib(validator=optional(instance_of(np.ndarray)), default=None)


@attr.s(cmp=False)
class Vocalization:
    """class to represent an annotated vocalization

    Attributes
    ----------
    annotation : list, tuple, or crowsetta.Sequence
        annotations of vocalizations for files
    audio_file : str
        path to file containing audio of vocalization
    audio : numpy.ndarray
        audio waveform loaded into a numpy array
    spect_file : str
        path to file containing spectrogram of vocalization as an array
    spect : vak.dataset.Spectrogram
        spectrogram of vocalization. Represented as an instance of the
        Spectrogram class, see docstring of that class for its attributes.
    """
    annotation = attr.ib()
    @annotation.validator
    def is_list_tup_or_seq(self, attribute, value):
        if type(value) not in (list, tuple, Sequence):
            raise TypeError(
                f'annotations for Vocalization must be a list, tuple, or crowsetta.Sequence'
            )
    audio_file = attr.ib(validator=instance_of(str), default=None)
    audio = attr.ib(validator=optional(instance_of(np.ndarray)), default=None)
    spect = attr.ib(validator=optional(instance_of(Spectrogram)), default=None)
    spect_file = attr.ib(validator=optional(instance_of(str)), default=None)


@attr.s(cmp=False)
class VocalDataset:
    """class to represent a dataset of annotated vocalizations

    Attributes
    ----------
    voc_list : list
        of Vocalizations.
    """
    voc_list = attr.ib()
    @voc_list.validator
    def all_voc(self, attribute, value):
        if not all([type(element) == Vocalization for element in value]):
            raise TypeError(f'all elements in voc_list must be of type vak.dataset.Vocalization')

    def to_json(self, json_fname=None):
        voc_dataset_dict = attr.asdict(self)
        if json_fname:
            with open(json_fname, 'w') as fp:
                json.dump(voc_dataset_dict, fp)
        else:
            return json.dumps(voc_dataset_dict)

    @classmethod
    def from_json(cls, json_fname):
        with open(json_fname, 'r') as fp:
            voc_dataset_dict = json.load(fp)
        voc_list = voc_dataset_dict['voc_list']
        voc_list = [Vocalization(**voc) for voc in voc_list]
        voc_dataset = cls(voc_list=voc_list)
        return voc_dataset

    def save(self, json_fname):
        self.to_json(json_fname)

    def load(self, json_fname):
        return self.from_json(json_fname)
