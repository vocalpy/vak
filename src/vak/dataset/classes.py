import json
from json import JSONEncoder

import numpy as np
from scipy.io import loadmat
import dask.bag as db
from dask.diagnostics import ProgressBar
import attr
from attr.validators import optional, instance_of
from crowsetta import Sequence

from ..utils.general import timebin_dur_from_vec


def asarray_if_not(val):
    if val is None:
        return None
    else:
        if type(val) == np.ndarray:
            return val
        else:
            return np.asarray(val)


@attr.s(cmp=False)
class SpectrogramFile:
    """class to represent a spectrogram file, e.g. a .mat or .npz file that
    contains the spectrogram and associated arrays

    Attributes
    ----------
    freq_bins : numpy.ndarray
        vector of frequencies in spectrogram, where each value is a bin center.
    time_bins : numpy.ndarray
        vector of times in spectrogram, where each value is a bin center.
    timebin_dur : numpy.ndarray
        duration of a timebin in seconds from spectrogram
    spect : numpy.ndarray
        spectrogram contained in an array
    """
    freq_bins = attr.ib(validator=instance_of(np.ndarray), converter=asarray_if_not)
    time_bins = attr.ib(validator=instance_of(np.ndarray), converter=asarray_if_not)
    timebin_dur = attr.ib(validator=instance_of(float))
    spect = attr.ib(validator=optional(instance_of(np.ndarray)), converter=asarray_if_not)

    @classmethod
    def from_arr_file_dict(cls,
                           arr_file_dict,
                           freqbins_key='f',
                           timebins_key='t',
                           spect_key='s',
                           timebin_dur=None,
                           n_decimals_trunc=3):
        """create a Spectrogram instance from a dictionary-like object that
        provides access to arrays loaded from a file, e.g. a .mat or .npz file

        Parameters
        ----------
        arr_file_dict : dict-like
            dictionary-like object providing access to .mat or .npz file
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
        spect : vak.dataset.classes.SpectrogramFile
            a Spectrogram instance with attributes freq_bins, time_bins, array, and timebin_dur
        """
        if timebin_dur is None:
            timebin_dur = timebin_dur_from_vec(time_bins=arr_file_dict[timebins_key],
                                               n_decimals_trunc=n_decimals_trunc)

        return cls(freq_bins=arr_file_dict[freqbins_key],
                   time_bins=arr_file_dict[timebins_key],
                   array=arr_file_dict[spect_key],
                   timebin_dur=timebin_dur)


def voc_file_validator(instance, attribute, value):
    if ((attribute.name == 'audio_file' and value is None) and (instance.spect_file is None) or
            (attribute.name == 'spect_file' and value is None) and (instance.audio_file is None)):
        raise ValueError(
            'a vocalization must have either an audio_file or spect_file associated with it'
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
    spect : vak.dataset.Spectrogram
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
    @annot.validator
    def is_list_tup_or_seq(self, attribute, value):
        if type(value) not in (list, tuple, Sequence):
            raise TypeError(
                f'annotations for Vocalization must be a crowsetta.Sequence'
            )
    # optional: need *one of* audio_file + audio or spect + spect_file
    audio = attr.ib(validator=optional(instance_of(np.ndarray)),
                    converter=asarray_if_not,
                    default=None)
    audio_file = attr.ib(validator=[optional(instance_of(str)), voc_file_validator],
                         default=None)
    spect = attr.ib(validator=optional(instance_of(SpectrogramFile)),
                    default=None)
    spect_file = attr.ib(validator=[optional(instance_of(str)), voc_file_validator],
                         default=None)


class VocalDatasetJSONEncoder(JSONEncoder):
    def default(self, o):
        if type(o) == Sequence:
            return o.as_dict()
        elif type(o) == np.ndarray:
            return o.tolist()
        else:
            return json.JSONEncoder.default(self, o)


@attr.s(cmp=False)
class VocalizationDataset:
    """class to represent a dataset of annotated vocalizations

    Attributes
    ----------
    voc_list : list
        of Vocalizations.
    """
    voc_list = attr.ib()

    @voc_list.validator
    def is_list_or_tuple(self, attribute, value):
        if type(value) not in (list, tuple):
            raise TypeError(
                f'{attribute.name} must be either a list or tuple'
            )

    @voc_list.validator
    def all_voc(self, attribute, value):
        if not all([type(element) == Vocalization for element in value]):
            raise TypeError(f'all elements in voc_list must be of type vak.dataset.Vocalization')

    def load_spects(self,
                    freqbins_key='f',
                    timebins_key='t',
                    spect_key='s',
                    n_decimals_trunc=3,
                    ):
        if not all(
            [hasattr(voc, 'spect_file') for voc in self.voc_list]
        ):
            raise ValueError(
                "not all Vocalizations in voc_list have a spect_file attribute, "
                "can't load spectrogram arrays"
            )

        if all([voc.spect_file.endswith('.mat') for voc in self.voc_list]):
            ext = 'mat'
        elif all([voc.spect_file.endswith('.npz') for voc in self.voc_list]):
            ext = 'npz'
        else:
            ext_set = [voc.spect_file.split[-1] for voc in self.voc_list]
            ext_set = set(ext_set)
            raise ValueError(
                f"unable to load spectrogram files, found multiple extensions: {ext_set}"
            )

        def _load_spect(voc):
            """helper function to load spectrogram into a Vocalization"""
            if ext == 'mat':
                arr_dict = loadmat(voc.spect_file)
            elif ext == 'npz':
                arr_dict = np.load(voc.spect_file)
            spect_dict = {
                'freq_bins': arr_dict[freqbins_key],
                'time_bins': arr_dict[timebins_key],
                'timebin_dur': timebin_dur_from_vec(arr_dict[timebins_key], n_decimals_trunc),
                'array': arr_dict[spect_key],
            }
            spect = SpectrogramFile(**spect_dict)
            voc.spect = spect
            return voc

        voc_db = db.from_sequence(self.voc_list)
        with ProgressBar():
            self.voc_list = list(voc_db.map(_load_spect))

    def clear_spects(self):
        for voc in self.voc_list:
            voc.spect = None

    def are_spects_loaded(self):
        if all([voc.spect is None for voc in self.voc_list]):
            return False
        elif all([type(voc.spect == SpectrogramFile) for voc in self.voc_list]):
            return True
        else:
            raise ValueError(
                """Not all Vocalizations in voc_list have spectrograms loaded. 
                Call load_spects() to load all or clear_spects() to set them all to None"""
            )

    def spects_list(self, load=True, load_kwargs=None):
        """returns list of spectrograms (2-d arrays),
        one for each vocalization in VocalizationDataset.voc_list"""
        if self.are_spects_loaded() is False:
            if load is True:
                self.load_spects(**load_kwargs)
            elif load is False:
                raise ValueError('cannot create list of spectrograms, because they are not loaded '
                                 'and load is set to False. Either call load_spects method or set '
                                 'load=True when calling spects_list')

        spects = []
        for voc in self.voc_list:
            spects.append(voc.spect.array)
        return spects

    def labels_list(self):
        """returns list of labels from annotations,
        one for each vocalization in VocalizationDataset.voc_list"""
        labels = []
        for voc in self.voc_list:
            labels.append(voc.annot.labels)
        return labels

    def to_json(self, json_fname=None):
        voc_dataset_dict = attr.asdict(self)
        if json_fname:
            with open(json_fname, 'w') as fp:
                json.dump(voc_dataset_dict, fp, cls=VocalDatasetJSONEncoder)
        else:
            return json.dumps(voc_dataset_dict, cls=VocalDatasetJSONEncoder)

    @classmethod
    def from_json(cls, json_str=None, json_fname=None):
        if json_str is None and json_fname is None:
            raise ValueError('must supply either json_str or json_fname argument')
        elif json_str and json_fname:
            raise ValueError('cannot provide both json_str and json_fname argument, '
                             'unclear which to convert back to VocalDataset')

        if json_fname:
            with open(json_fname, 'r') as fp:
                voc_dataset_dict = json.load(fp)
        elif json_str:
            voc_dataset_dict = json.loads(json_str)

        for a_voc_dict in voc_dataset_dict['voc_list']:
            if a_voc_dict['annot'] is not None:
                a_voc_dict['annot'] = Sequence.from_dict(a_voc_dict['annot'])
            if a_voc_dict['spect'] is not None:
                a_voc_dict['spect'] = SpectrogramFile(**a_voc_dict['spect'])

        voc_dataset_dict['voc_list'] = [Vocalization(**voc) for voc in voc_dataset_dict['voc_list']]

        return cls(voc_list=voc_dataset_dict['voc_list'])

    def save(self, json_fname):
        self.to_json(json_fname)

    @classmethod
    def load(cls, json_fname):
        return cls.from_json(json_fname=json_fname)
