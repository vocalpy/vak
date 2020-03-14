import os
from pathlib import Path

import crowsetta

from ..config import validators
from .general import _files_from_dir

NO_ANNOTATION_FORMAT = 'none'


def format_from_df(vak_df):
    """determine annotation format of a Vak DataFrame.
    Returns string name of annotation format.

    Raises an error if there is no annotation format, or multiple formats.

    Parameters
    ----------
    vak_df : DataFrame
        representating a dataset of vocalizations, with column 'annot_format'.

    Returns
    -------
    annot_format : str
        format of annotations for vocalizations.
    """
    annot_format = vak_df['annot_format'].unique()
    if len(annot_format) == 1:
        annot_format = annot_format.item()
        # if annot_format is None, throw an error -- otherwise continue on and try to use it
        if annot_format is None:
            raise ValueError(
                'unable to load labels for dataset, the annot_format is None'
            )
        elif annot_format is NO_ANNOTATION_FORMAT:
            raise ValueError(
                'unable to load labels for dataset, no annotation format is specified'
            )
    elif len(annot_format) > 1:
        raise ValueError(
            f'unable to load labels for dataset, found multiple annotation formats: {annot_format}'
        )

    return annot_format


def from_df(vak_df):
    """get list of annotations from a vak DataFrame

    Parameters
    ----------
    vak_df : DataFrame
        representating a dataset of vocalizations, with column 'annot_format'.

    Returns
    -------
    annots : list
        of annotations for each row in the dataframe,
        represented as crowsetta.Annotation instances.
    """
    annot_format = format_from_df(vak_df)

    scribe = crowsetta.Transcriber(annot_format=annot_format)

    if len(vak_df['annot_path'].unique()) == len(vak_df):
        # --> there is a unique annotation file (path) for each row, iterate over them to get labels from each
        annots = [scribe.from_file(annot_file=annot_path) for annot_path in vak_df['annot_path'].values]

    elif len(vak_df['annot_path'].unique()) == 1:
        # --> there is a single annotation file associated with all rows
        annot_path = vak_df['annot_path'].unique().item()
        annots = scribe.from_file(annot_file=annot_path)

        if type(annots) == list and len(annots) == len(vak_df):
            audio_annot_map = source_annot_map(vak_df['audio_path'].values, annots)
            # sort by row of dataframe
            annots = [audio_annot_map[audio_path] for audio_path in vak_df['audio_path'].values]

        else:
            raise ValueError(
                'unable to load labels from dataframe; found a single annotation file associated with all '
                'rows in dataframe, but loading it did not return a list of annotations for each row.\n'
                f'Single annotation file: {annot_path}'
                f'Loading it returned a {type(annots)}.'
            )
    else:
        raise ValueError(
            'unable to load labels from dataframe; did not find an annotation file for each row or '
            'a single annotation file associated with all rows.'
        )

    return annots


def files_from_dir(annot_dir, annot_format):
    """get all annotation files of a given format
    from a directory or its sub-directories,
    using the file extension associated with that annotation format.
    """
    if annot_format not in validators.VALID_ANNOT_FORMATS:
        raise ValueError(
            f'specified annotation format, {annot_format} not valid.\n'
            f'Valid formats are: {validators.VALID_ANNOT_FORMATS}'
        )

    format_module = getattr(crowsetta.formats, annot_format)
    ext = format_module.meta.ext
    annot_files = _files_from_dir(annot_dir, ext)
    return annot_files


def _recursive_stem(path_str):
    name = Path(path_str).name
    stem, ext = os.path.splitext(name)
    ext = ext.replace('.', '')
    while ext not in validators.VALID_AUDIO_FORMATS:
        new_stem, ext = os.path.splitext(stem)
        ext = ext.replace('.', '')
        if new_stem == stem:
            raise ValueError(
                f'unable to compute stem of {path_str}'
            )
        else:
            stem = new_stem
    return stem


def source_annot_map(source_files, annot_list):
    """map annotations to their source files, i.e. audio or spectrogram files

    Parameters
    ----------
    source_files : list
        of audio or spectrogram files. The names of the files must match the
        file attribute of the annotations. E.g., if an audio file is
        'bird0-2016-05-04-133027.wav', then there must be an annotation whose
        file attribute equals that filename. Spectrogram files should include
        the audio file name, e.g. 'bird0-2016-05-04-133027.wav.mat' or
        'bird0-2016-05-04-133027.spect.npz'
    annot_list : list
        of Annotations corresponding to files in source_files
    """
    # to pair audio files with annotations, make list of tuples
    source_annot_map = []  # that we convert a dict before returning
    # we copy source files so we can convert to stem, and so we can
    # pop items off copy without losing original list.
    # We pop to validate that function worked, by making sure there are
    # no items left in this list after the loop
    source_files_stem = source_files.copy()
    source_files_stem = [_recursive_stem(sf) for sf in source_files_stem]
    source_file_inds = list(range(len(source_files)))
    for annot in annot_list:
        # remove stem so we can find .spect files that match with audio files,
        # e.g. find 'llb3_0003_2018_04_23_14_18_54.mat' that should match
        # with 'llb3_0003_2018_04_23_14_18_54.wav'
        annot_file_stem = _recursive_stem(annot.audio_file)

        ind_in_stem = [ind
                       for ind, source_file_stem in enumerate(source_files_stem)
                       if annot_file_stem == source_file_stem]
        if len(ind_in_stem) > 1:
            more_than_one = [source_file_inds[ind] for ind in ind_in_stem]
            more_than_one = [source_files[ind] for ind in more_than_one]
            raise ValueError(
                "Found more than one source file that matches an annotation."
                f"\nSource files are: {more_than_one}."
                f"\nAnnotation has file set to '{annot.audio_file}' and is: {annot}"
            )
        elif len(ind_in_stem) == 0:
            raise ValueError(
                "Did not find a source file matching the following annotation: "
                f"\n{annot}. Annotation has file set to '{annot.audio_file}'."
            )
        else:
            ind_in_stem = ind_in_stem[0]
            ind = source_file_inds[ind_in_stem]
            source_annot_map.append(
                (source_files[ind], annot)
            )
            source_files_stem.pop(ind_in_stem)
            source_file_inds.pop(ind_in_stem)

    if len(source_files_stem) > 0:
        raise ValueError(
            'could not map the following source files to annotations: '
            f'{source_files_stem}'
        )

    return dict(source_annot_map)
