import json
from json import JSONEncoder

import numpy as np
from scipy.io import loadmat
import dask.bag as db
from dask.diagnostics import ProgressBar
import attr
from attr.validators import optional, instance_of
from crowsetta import Sequence

from .vocalization import Vocalization
from .metaspect import MetaSpect
from ...utils.general import timebin_dur_from_vec
from ...utils.labels import label_timebins


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
    labelset : set
        (unique) set of labels used in annotations for all Vocalizations in data set
    labelmap : dict
        dictionary that maps labelset to consecutive integer values {0,1,2,...N} where N
        is the number of classes / label types
    """
    voc_list = attr.ib(validator=instance_of(list))
    labelset = attr.ib(validator=optional(instance_of(set)), default=None)
    labelmap = attr.ib(validator=optional(instance_of(dict)), default=None)

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
            [hasattr(voc, 'spect_path') for voc in self.voc_list]
        ):
            raise ValueError(
                "not all Vocalizations in voc_list have a spect_path attribute, "
                "can't load spectrogram arrays"
            )

        if all([voc.spect_path.endswith('.mat') for voc in self.voc_list]):
            ext = 'mat'
        elif all([voc.spect_path.endswith('.npz') for voc in self.voc_list]):
            ext = 'npz'
        else:
            ext_set = [voc.spect_path.split[-1] for voc in self.voc_list]
            ext_set = set(ext_set)
            raise ValueError(
                f"unable to load spectrogram files, found multiple extensions: {ext_set}"
            )

        def _load_spect(voc):
            """helper function to load spectrogram into a Vocalization"""
            if ext == 'mat':
                spect_dict = loadmat(voc.spect_path)
            elif ext == 'npz':
                spect_dict = np.load(voc.spect_path)
            metaspect_kwargs = {
                'freq_bins': spect_dict[freqbins_key],
                'time_bins': spect_dict[timebins_key],
                'timebin_dur': timebin_dur_from_vec(spect_dict[timebins_key], n_decimals_trunc),
                'spect': spect_dict[spect_key],
            }
            voc.metaspect = MetaSpect(**metaspect_kwargs)
            return voc

        voc_db = db.from_sequence(self.voc_list)
        with ProgressBar():
            self.voc_list = list(voc_db.map(_load_spect))

    def clear_spects(self):
        """sets Vocalization.metaspect to None, for every Vocalization in VocaliationDataset.voc_list

        Useful for clearing arrays from the VocalizationDataset before saving; going to and from .json
        with numpy.ndarrays loaded into MetaSpect attributes can be very slow.
        """
        for voc in self.voc_list:
            voc.metaspect = None

    def are_spects_loaded(self):
        """reports whether spectrogram files have been loaded into MetaSpect instances

        Returns
        -------
        spects_loaded : bool
            True if all Vocalizations in a VocalizationDataset have a metaspect.
            False if instead the metaspect attribute for all Vocalizations is set to None.
        """
        if all([voc.metaspect is None for voc in self.voc_list]):
            return False
        elif all([type(voc.metaspect == MetaSpect) for voc in self.voc_list]):
            return True
        else:
            raise ValueError(
                """Not all Vocalizations in voc_list have spectrograms loaded. 
                Call load_spects() to load all or clear_spects() to set them all to None"""
            )

    def spects_list(self, load=True, load_kwargs=None):
        """returns list of spectrograms (2-d arrays),
        one for each vocalization in VocalizationDataset.voc_list

        Parameters
        ----------
        load
        load_kwargs

        Returns
        -------
        spects_list : list
            of Vocalization.metaspect.spect, one for each Vocalization in the VocalizationDataset.
            Each element is a spectrogram in a numpy.ndarray.
        """
        if self.are_spects_loaded() is False:
            if load is True:
                self.load_spects(**load_kwargs)
            elif load is False:
                raise ValueError('cannot create list of spectrograms, because they are not loaded '
                                 'and load is set to False. Either call load_spects method or set '
                                 'load=True when calling spects_list')

        spects_list = []
        for voc in self.voc_list:
            spects_list.append(voc.metaspect.spect)
        return spects_list

    def labels_list(self):
        """returns list of labels from annotations,
        one for each vocalization in VocalizationDataset.voc_list

        Returns
        -------
        labels_list : list
            of Vocalization.annot.labels, one for each Vocalization in the VocalizationDataset.
            Each element of the list is a numpy.ndarray.
        """
        labels_list = []
        for voc in self.voc_list:
            labels_list.append(voc.annot.labels)
        return labels_list

    def lbl_tb_list(self):
        """returns list of labeled time bin vectors from annotations,
        one for each vocalization in VocalizationDataset.voc_list

        Returns
        -------
        lbl_tb_list : list
            that results from applying utils.labels.label_timebins to each Vocalization
            in the VocalizationDataset.
        """
        if self.labelmap is None:
            raise ValueError(
                'this VocalizationDataset has no labelmap; please create a labelmap from '
                'the labelset and assign it to the labelmap attribute'
            )

        if 'unlabeled' in self.labelmap:
            unlabeled_label = self.labelmap['unlabeled']
        else:
            # if there is no "unlabeled label" (e.g., because all segments have labels)
            # just assign dummy value that will be replaced by some label in label_timebins()
            unlabeled_label = 0

        lbl_tb_list = []
        for voc in self.voc_list:
            lbls_int = [self.labelmap[lbl] for lbl in voc.annot.labels]
            lbl_tb_list.append(
                label_timebins(lbls_int,
                               voc.annot.onsets,
                               voc.annot.offsets,
                               voc.spect.time_bins,
                               unlabeled_label=unlabeled_label)
            )
        return lbl_tb_list

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
            if a_voc_dict['metaspect'] is not None:
                a_voc_dict['metaspect'] = MetaSpect(**a_voc_dict['metaspect'])

        voc_dataset_dict['voc_list'] = [Vocalization(**voc) for voc in voc_dataset_dict['voc_list']]

        return cls(voc_list=voc_dataset_dict['voc_list'])

    def save(self, json_fname):
        self.to_json(json_fname)

    @classmethod
    def load(cls, json_fname):
        return cls.from_json(json_fname=json_fname)
