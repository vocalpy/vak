import numpy as np
import attr
from attr.validators import instance_of, optional
from crowsetta import Sequence

from .metaspect import MetaSpect
from .validators import asarray_if_not


def voc_path_validator(instance, attribute, value):
    if ((attribute.name == 'audio_path' and value is None) and (instance.spect_path is None) or
            (attribute.name == 'spect_path' and value is None) and (instance.audio_path is None)):
        raise ValueError(
            'a vocalization must have either an audio_path or spect_path associated with it'
            )


@attr.s(cmp=False)
class Vocalization:
    """class to represent an annotated vocalization

    Attributes
    ----------
    annot : crowsetta.Sequence
        annotations of vocalizations for files
    audio_file : str
        path to file containing audio of vocalization
    audio : numpy.ndarray
        audio waveform loaded into a numpy array
    spect_file : str
        path to file containing spectrogram of vocalization as an array
    spect : vak.dataset.classes.MetaSpect
        spectrogram of vocalization. Represented as an instance of the
        Spectrogram class, see docstring of that class for its attributes.
    duration : float
        duration of audio file and/or spectrogram associated with vocalization
    """
    # mandatory attribute: duration
    duration = attr.ib(validator=instance_of(float))

    # less mandatory: annotation
    # need at least labels from annotation + duration to split a dataset up by duration
    # while maintaining class balance
    annot = attr.ib(validator=optional(instance_of(Sequence)), default=None)

    # optional: need *one of* audio_file + audio or spect + metaspect
    audio = attr.ib(validator=optional(instance_of(np.ndarray)),
                    converter=asarray_if_not,
                    default=None)
    audio_path = attr.ib(validator=[optional(instance_of(str)), voc_path_validator],
                         default=None)
    metaspect = attr.ib(validator=optional(instance_of(MetaSpect)),
                        default=None)
    spect_path = attr.ib(validator=[optional(instance_of(str)), voc_path_validator],
                         default=None)
