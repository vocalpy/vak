from __future__ import annotations
from collections import Counter
import copy
import os
from pathlib import Path

import crowsetta
import numpy as np

from . import files
from . import constants


def format_from_df(vak_df):
    """determine annotation format of a Vak DataFrame.
    Returns string name of annotation format.
    If no annotation format is specified, returns None.
    Raises an error if there are multiple formats.

    Parameters
    ----------
    vak_df : pandas.DataFrame
        Representing a dataset of vocalizations,
        with column 'annot_format'.

    Returns
    -------
    annot_format : str
        format of annotations for vocalizations.
    """
    annot_format = vak_df["annot_format"].unique()
    if len(annot_format) == 1:
        annot_format = annot_format.item()
        if annot_format is None or annot_format == constants.NO_ANNOTATION_FORMAT:
            return None
    elif len(annot_format) > 1:
        raise ValueError(
            f"unable to load labels for dataset, found multiple annotation formats: {annot_format}"
        )

    return annot_format


def from_df(vak_df):
    """get list of annotations from a vak DataFrame.
    If no annotation format is specified for the DataFrame
    (in the 'annot_format' column), returns None.

    Parameters
    ----------
    vak_df : DataFrame
        representating a dataset of vocalizations, with column 'annot_format'.

    Returns
    -------
    annots : list
        of annotations for each row in the dataframe,
        represented as crowsetta.Annotation instances.

    Notes
    -----
    This function encapsulates logic for handling different types of
    annotations; it determines whether each row has a separate annotation file,
    or if instead there is a single annotation file associated with all rows.
    If the latter, then the function opens that file and makes sure that
    each row from the dataframe can be paired with an annotation (using `map_annotated_to_annot`).
    """
    annot_format = format_from_df(vak_df)
    if annot_format is None:
        return None

    scribe = crowsetta.Transcriber(format=annot_format)

    if len(vak_df["annot_path"].unique()) == 1:
        # --> there is a single annotation file associated with all rows
        # this can be true in two different cases:
        # (1) many rows, all have the same file
        # (2) only one row, so there's only one annotation file (which may contain annotation for multiple source files)
        annot_path = vak_df["annot_path"].unique().item()
        annots = scribe.from_file(annot_path)

        # as long as we have at least as many annotations as there are rows in the dataframe
        if (isinstance(annots, list) and len(annots) >= len(vak_df)) or (  # case 1
            isinstance(annots, crowsetta.Annotation) and len(vak_df) == 1
        ):  # case 2
            if isinstance(annots, crowsetta.Annotation):
                annots = [
                    annots
                ]  # wrap in list for map_annotated_to_annot to iterate over it
            # then we can try and map those annotations to the rows
            audio_annot_map = map_annotated_to_annot(vak_df["audio_path"].values, annots)
            # sort by row of dataframe
            annots = [
                audio_annot_map[audio_path]
                for audio_path in vak_df["audio_path"].values
            ]

        else:
            raise ValueError(
                "unable to load labels from dataframe; found a single annotation file associated with all "
                "rows in dataframe, but loading it did not return a list of annotations for each row.\n"
                f"Single annotation file: {annot_path}\n"
                f"Loading it returned a {type(annots)}."
            )

    elif len(vak_df["annot_path"].unique()) == len(vak_df):
        # --> there is a unique annotation file (path) for each row, iterate over them to get labels from each
        annots = [
            scribe.from_file(annot_path) for annot_path in vak_df["annot_path"].values
        ]

    else:
        raise ValueError(
            "unable to load labels from dataframe; did not find an annotation file for each row or "
            "a single annotation file associated with all rows."
        )

    return annots


def files_from_dir(annot_dir, annot_format):
    """get all annotation files of a given format
    from a directory or its sub-directories,
    using the file extension associated with that annotation format.
    """
    if annot_format not in constants.VALID_ANNOT_FORMATS:
        raise ValueError(
            f"specified annotation format, {annot_format} not valid.\n"
            f"Valid formats are: {constants.VALID_ANNOT_FORMATS}"
        )

    format_module = getattr(crowsetta.formats, annot_format)
    ext = format_module.meta.ext
    annot_files = files.from_dir(annot_dir, ext)
    return annot_files


class AudioFilenameNotFound(Exception):
    """Error raised by ``audio_stem_from_path``"""


def audio_stem_from_path(path):
    """Find the name of an audio file within a filename
    by removing extensions until finding an audio extension,
    then return the name of that audio file
    without the extension (i.e., the "stem").

    Removes extensions from a filename recursively,
    by calling `os.path.splitext`,
    until the extension is an audio file format handled by vak.
    Then return the stem, that is, the part that precedes the extension.
    Used to match audio, spectrogram, and annotation files by their stems.

    Stops after finding audio extensions so that it does not remove "extensions"
    that are actually other parts of a filename, e.g. a time or data separated by periods.

    Examples
    --------
    >>> audio_stem_from_path('gy6or6_baseline_230312_0808.138.cbin.not.mat')
    'gy6or6_baseline_230312_0808.138'
    >>> audio_stem_from_path('Bird0/spectrograms/0.wav.npz')
    '0'
    >>> audio_stem_from_path('Bird0/Wave/0.wav')
    '0'

    Parameters
    ----------
    path : str, Path
        from which stem should be extracted

    Returns
    -------
    stem : str
        filename that precedes audio extension
    """
    name = Path(path).name
    stem, ext = os.path.splitext(name)
    ext = ext.replace(".", "").lower()
    while ext not in constants.VALID_AUDIO_FORMATS:
        new_stem, ext = os.path.splitext(stem)
        ext = ext.replace(".", "").lower()
        if new_stem == stem:
            raise AudioFilenameNotFound(
                f"Unable to find a valid audio filename in path:\n{path}.\n"
                f"Valid audio file extensions are:\n{constants.VALID_AUDIO_FORMATS}"
            )
        else:
            stem = new_stem
    return stem


def map_annotated_to_annot(annotated_files: list,
                           annot_list: crowsetta.Annotation) -> dict:
    """Map annotated files,
    i.e. audio or spectrogram files,
    to their corresponding annotations.

    Returns a ``dict`` where each key
    is a path to an annotated file,
    and the value for each key
    is a ``crowsetta.Annotation``.

    Parameters
    ----------
    annotated_files : list
        Of paths to audio or spectrogram files.
    annot_list : list
        of Annotations corresponding to files in annotated_files

    Notes
    -----
    The filenames of the ``annotated_files`` must
    begin with the filename of the ``audio_path``
    attribute of the corresponding
    ``crowsetta.Annotation`` instances.
    E.g., if `annotated_files` includes
    an audio file named
    'bird0-2016-05-04-133027.wav',
    then it will be mapped to an ``Annotation``
    with an `audio_path` attribute
    whose filename matches it.
    Spectrogram files should also include
    the audio file name,
    e.g. 'bird0-2016-05-04-133027.wav.mat'
    or 'bird0-2016-05-04-133027.spect.npz'
    would match an ``Annotation`` with the
    ``audio_path`` attribute '/some/path/bird0-2016-05-04-133027.wav'.

    For more detail, please see
    the page on file naming conventions in the
    reference section of the documentation:
    https://vak.readthedocs.io/en/latest/reference/filenames.html
    """
    if type(annotated_files) == np.ndarray:  # e.g., vak DataFrame['spect_path'].values
        annotated_files = annotated_files.tolist()

    # to pair audio files with annotations, make list of tuples
    annotated_annot_map = {}

    # ----> make a dict with audio stems as keys,
    #       so we can look up annotations by stemming source files and using as keys.
    # First check that we don't have duplicate keys that would cause this to fail silently
    keys = []
    for annot in annot_list:
        try:
            key = audio_stem_from_path(annot.audio_path)
        except AudioFilenameNotFound as e:
            # Do this as a loop with a super verbose error
            # instead of e.g. a single-line list comprehension
            # so we can help users troubleshoot,
            # see https://github.com/vocalpy/vak/issues/525
            raise ValueError(
                "The ``audio_path`` attribute of a ``crowsetta.Annotation`` was "
                "not recognized as a valid audio filename.\n"
                f"The ``audio_path`` attribute was:\n{annot.audio_path}\n"
                f"The annotation was loaded from this path:\n{annot.annot_path}\n"
                "For some annotation formats, audio filenames are inferred from annotation filenames.\n"
                "Please check that your annotation files are named "
                "according to the conventions:\n"
                "https://vak.readthedocs.io/en/latest/reference/filenames.html\n"
                "It may also be helpful to read the page on converting custom formats "
                "to annotations that ``vak`` can work with:\n"
                "https://vak.readthedocs.io/en/latest/howto/howto_user_annot.html"
            ) from e
        keys.append(key)

    keys_set = set(keys)
    if len(keys_set) < len(keys):
        duplicates = [item for item, count in Counter(keys).items() if count > 1]
        raise ValueError(
            f"found multiple annotations with the same audio filename(s): {duplicates}"
        )
    del keys, keys_set
    audio_stem_annot_map = {
        audio_stem_from_path(annot.audio_path): annot for annot in annot_list
    }

    # Make a copy from which we remove source files after mapping them to annotation,
    # to validate that function worked,
    # by making sure there are no items left in this copy after the loop
    annotated_files_copy = copy.deepcopy(annotated_files)
    for annotated_file in list(
        annotated_files
    ):  # list() to copy, so we can pop off items while iterating
        # remove stem so we can find .spect files that match with audio files,
        # e.g. find 'llb3_0003_2018_04_23_14_18_54.mat' that should match
        # with 'llb3_0003_2018_04_23_14_18_54.wav'
        annotated_file_stem = audio_stem_from_path(annotated_file)
        annot = audio_stem_annot_map[annotated_file_stem]
        annotated_annot_map[annotated_file] = annot
        annotated_files_copy.remove(annotated_file)

    if len(annotated_files_copy) > 0:
        raise ValueError(
            "could not map the following source files to annotations: "
            f"{annotated_files_copy}"
        )

    return annotated_annot_map


def has_unlabeled(annot: crowsetta.Annotation,
                  duration: float):
    """Returns ``True`` if an annotated sequence has unlabeled segments.

    Tests whether an instance of ``crowsetta.Annotation.seq`` has
    intervals between the annotated segments with a non-zero duration,
    or any unannotated periods before or after the annotated segments.

    Parameters
    ----------
    annot : crowsetta.Annotation
        With a ``seq`` attribute that is a ``crowsetta.Sequence``
    duration : float
        Total duration of the vocalization
        that is annotated by ``annot``.
        Needed to determine whether the duration
        is greater than the time
        of the last offset in the annotated segments.

    Returns
    -------
    has_unlabeled : bool
        If True, there are unlabeled periods
        in the vocalization annotated by ``annot``.
    """
    has_unlabeled_intervals = np.any((annot.seq.onsets_s[1:] - annot.seq.offsets_s[:-1]) > 0.)
    has_unlabeled_before_first_onset = annot.seq.onsets_s[0] > 0.
    has_unlabeled_after_last_offset = duration - annot.seq.offsets_s[-1] > 0.
    return has_unlabeled_intervals or has_unlabeled_before_first_onset or has_unlabeled_after_last_offset
