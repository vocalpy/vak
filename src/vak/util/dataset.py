import crowsetta
import pandas as pd

from . import labels
from .path import array_dict_from_path


def has_unlabeled(csv_path, labelset, timebins_key='t'):
    """determine if a dataset has segments that are unlabeled.

    Used to decide whether an additional class needs to be added
    to the set of labels Y = {y_1, y_2, ... y_n}, where the added
    class y_n+1 will represent the unlabeled segments.

    Parameters
    ----------
    csv_path : str, Path
        to .csv file representing dataset.
    labelset : set
        of labels, str or int, e.g. {'a', 'b', 'c'}
    timebins_key : str
        key used to access timebins vector in spectrogram files.
        Default is 't'.

    Returns
    -------
    has_unlabeled : bool
        if True, dataset has unlabeled segments.
    """
    tmp_labelmap = labels.to_map(labelset, map_unlabeled=False)

    vak_df = pd.read_csv(csv_path)
    uniq_annot_formats = vak_df['annot_format'].unique()
    if len(uniq_annot_formats) == 1:
        has_only_one_annot_format = True
        annot_format = uniq_annot_formats.item()
        scribe = crowsetta.Transcriber(annot_format=annot_format)
    else:
        has_only_one_annot_format = False

    has_unlabeled_list = []
    for _, voc in vak_df.iterrows():
        if not has_only_one_annot_format:
            scribe = crowsetta.Transcriber(annot_format=voc['annot_format'])
        annot = scribe.from_file(voc['annot_path'])
        time_bins = array_dict_from_path(voc['spect_path'])[timebins_key]

        lbls_int = [tmp_labelmap[lbl] for lbl in annot.seq.labels]
        has_unlabeled_list.append(
            labels.has_unlabeled(lbls_int,
                                 annot.seq.onsets_s,
                                 annot.seq.offsets_s,
                                 time_bins)
        )

    return any(has_unlabeled_list)
