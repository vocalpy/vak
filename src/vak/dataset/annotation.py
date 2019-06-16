import os
from pathlib import Path

import crowsetta

from ..config import validators
from ..utils.general import _files_from_dir


def files_from_dir(annot_dir, annot_format):
    """get all annotation files of a given format
    from a directory or its sub-directories,
    using the file extension associated with that annotation format.
    """
    if annot_format not in validators.VALID_ANNOT_FORMATS:
        raise ValueError(
            f'specified annotation format, {annot_format} not valid.\n'
            f'Valid formats are: {VALID_ANNOT_FORMATS}'
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
        annot_file_stem = _recursive_stem(annot.file)

        ind_in_stem = [ind
                       for ind, source_file_stem in enumerate(source_files_stem)
                       if annot_file_stem == source_file_stem]
        if len(ind_in_stem) > 1:
            more_than_one = [source_file_inds[ind] for ind in ind_in_stem]
            more_than_one = [source_files[ind] for ind in more_than_one]
            raise ValueError(
                "Found more than one source file that matches an annotation."
                f"\nSource files are: {more_than_one}."
                f"\nAnnotation has file set to '{annot.file}' and is: {annot}"
            )
        elif len(ind_in_stem) == 0:
            raise ValueError(
                "Did not find a source file matching the following annotation: "
                f"\n{annot}. Annotation has file set to '{annot.file}'."
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
            f'{audio_files}'
        )

    return dict(source_annot_map)
