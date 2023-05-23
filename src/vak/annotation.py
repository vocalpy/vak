from __future__ import annotations
from collections import Counter
import copy
import os
import pathlib
from typing import Optional, Union

import crowsetta
import numpy as np

from . import files
from . import constants
from .typing import PathLike


def format_from_df(dataset_df):
    """determine annotation format of a Vak DataFrame.
    Returns string name of annotation format.
    If no annotation format is specified, returns None.
    Raises an error if there are multiple formats.

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        Representing a dataset of vocalizations,
        with column 'annot_format'.

    Returns
    -------
    annot_format : str
        format of annotations for vocalizations.
    """
    annot_format = dataset_df["annot_format"].unique()
    if len(annot_format) == 1:
        annot_format = annot_format.item()
        if annot_format is None or annot_format == constants.NO_ANNOTATION_FORMAT:
            return None
    elif len(annot_format) > 1:
        raise ValueError(
            f"unable to load labels for dataset, found multiple annotation formats: {annot_format}"
        )

    return annot_format


def from_df(dataset_df):
    """get list of annotations from a vak DataFrame.
    If no annotation format is specified for the DataFrame
    (in the 'annot_format' column), returns None.

    Parameters
    ----------
    dataset_df : DataFrame
        representing a dataset of vocalizations, with column 'annot_format'.

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
    annot_format = format_from_df(dataset_df)
    if annot_format is None:
        return None

    scribe = crowsetta.Transcriber(format=annot_format)

    if len(dataset_df["annot_path"].unique()) == 1:
        # --> there is a single annotation file associated with all rows
        # this can be true in two different cases:
        # (1) many rows, all have the same file
        # (2) only one row, so there's only one annotation file (which may contain annotation for multiple source files)
        annot_path = dataset_df["annot_path"].unique().item()
        annots = scribe.from_file(annot_path).to_annot()

        # as long as we have at least as many annotations as there are rows in the dataframe
        if (isinstance(annots, list) and len(annots) >= len(dataset_df)) or (  # case 1
            isinstance(annots, crowsetta.Annotation) and len(dataset_df) == 1
        ):  # case 2
            if isinstance(annots, crowsetta.Annotation):
                annots = [
                    annots
                ]  # wrap in list for map_annotated_to_annot to iterate over it
            # then we can try and map those annotations to the rows
            audio_annot_map = map_annotated_to_annot(dataset_df["audio_path"].values, annots)
            # sort by row of dataframe
            annots = [
                audio_annot_map[audio_path]
                for audio_path in dataset_df["audio_path"].values
            ]

        else:
            raise ValueError(
                "unable to load labels from dataframe; found a single annotation file associated with all "
                "rows in dataframe, but loading it did not return a list of annotations for each row.\n"
                f"Single annotation file: {annot_path}\n"
                f"Loading it returned a {type(annots)}."
            )

    elif len(dataset_df["annot_path"].unique()) == len(dataset_df):
        # --> there is a unique annotation file (path) for each row, iterate over them to get labels from each
        annots = [
            scribe.from_file(annot_path).to_annot() for annot_path in dataset_df["annot_path"].values
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

    format_class = crowsetta.formats.by_name(annot_format)
    annot_files = files.from_dir(annot_dir, format_class.ext)
    return annot_files


class AudioFilenameNotFound(Exception):
    """Error raised when a name of an audio filename
    cannot be found within another filename.

    Raised by ``audio_filename_from_path``
    and ``_map_using_audio_stem_from_path``.
    """


def audio_filename_from_path(path: PathLike,
                             audio_ext: str = None) -> str:
    """Find the name of an audio file within a filename
    by removing extensions until finding an audio extension,
    then return the name of that audio file
    without the extension (i.e., the "stem").

    Removes extensions from a filename recursively,
    by calling `os.path.splitext`,
    until the extension is an audio file format handled by vak.
    Then return the stem, that is,
    the part that precedes the extension.
    Used to match audio, spectrogram,
    and annotation files by their stems.

    Stops after finding audio extensions
    so that it does not remove "extensions"
    that are actually other parts of a filename,
    e.g. a time or data separated by periods.

    Examples
    --------
    >>> audio_filename_from_path('gy6or6_baseline_230312_0808.138.cbin.not.mat')
    'gy6or6_baseline_230312_0808.138'
    >>> audio_filename_from_path('Bird0/spectrograms/0.wav.npz')
    '0'
    >>> audio_filename_from_path('Bird0/Wave/0.wav')
    '0'

    Parameters
    ----------
    path : str, Path
        Path to a file that contains an audio filename in its name.
    audio_ext : str
        Extension corresponding to format of audio file.
        Must be one of ``vak.constants.VALID_AUDIO_FORMATS``.
        Default is None, in which case the function looks
        removes extensions until it finds any valid audio
        format extension.

    Returns
    -------
    stem : str
        Part of filename that precedes audio extension.
    """
    if audio_ext:
        if audio_ext.startswith('.'):
            audio_ext = audio_ext[1:]
        if audio_ext not in constants.VALID_AUDIO_FORMATS:
            raise ValueError(
                f'Not a valid extension for audio formats: {audio_ext}\n'
                f'Valid formats are: {constants.VALID_AUDIO_FORMATS}'
            )
        extensions_to_look_for = [audio_ext]
    else:
        extensions_to_look_for = constants.VALID_AUDIO_FORMATS

    name = pathlib.Path(path).name
    stem, ext = os.path.splitext(name)
    ext = ext.replace(".", "").lower()
    while ext not in extensions_to_look_for:
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


class MapUsingNotatedPathError(BaseException):
    """Error raised when :func:`vak.annotation._map_using_notated_path`
    cannot map the filename of an annotation file to the name
    of an annotated file"""
    pass


def _map_using_notated_path(annotated_files: list[PathLike],
                            annot_list: list[crowsetta.Annotation],
                            audio_ext: Optional[str] = None) -> dict:
    """Map a :class:`list` of annotated files to a :class:`list`
    of  :class:`crowsetta.Annotation` instances,
    using the ``notated_path`` attribute of the
    :class:`~crowsetta.Annotation`.

    This function assumes that the annotation format
    includes the names of the files that it annotates.
    This is necessarily true for any format that puts
    annotations for multiple annotated files into a single
    annotation file.

    One of three helper functions used by 
    :func:`~vak.annotation.map_annotated_to_annot`.

    Parameters
    ----------
    annotated_files : list
        List of paths to the annotated files.
    annot_list : list
        List of ``crowsetta.Annotation`` instances.
    audio_ext : str
        Extension corresponding to audio format.
        Valid extension are listed in
        ``vak.constants.VALID_AUDIO_FORMATS``.
        Default is None, in which case the function
        looks for any valid format.

    Returns
    -------
    annotated_annot_map : dict
        Where each key is path to annotated file, and
        its value is the corresponding ``crowsetta.Annotation``.
    """
    # First check that we don't have duplicate keys that would cause this to fail silently
    keys = []
    for annot in annot_list:
        try:
            stem = audio_filename_from_path(annot.notated_path, audio_ext)
        except AudioFilenameNotFound as e:
            # Do this as a loop with a super verbose error
            # instead of e.g. a single-line list comprehension
            # so we can help users troubleshoot,
            # see https://github.com/vocalpy/vak/issues/525
            raise MapUsingNotatedPathError(
                "The ``notated_path`` attribute of a ``crowsetta.Annotation`` was "
                "not recognized as a valid audio filename.\n"
                f"The ``notated_path`` attribute was:\n{annot.notated_path}\n"
                f"The annotation was loaded from this path:\n{annot.annot_path}\n"
            ) from e
        keys.append(stem)

    keys_set = set(keys)
    if len(keys_set) < len(keys):
        duplicates = [item for item, count in Counter(keys).items() if count > 1]
        raise ValueError(
            f"found multiple annotations with the same audio filename(s): {duplicates}"
        )
    del keys, keys_set
    # ----> make a dict with audio stems as keys,
    #       so we can look up annotations
    #       by stemming source files and using as keys.
    audio_stem_annot_map = {
        # NOTE HERE WE GET STEMS FROM EACH annot.notated_path,
        # BELOW we get stems from each annotated_file
        audio_filename_from_path(annot.notated_path): annot for annot in annot_list
    }

    # Make a copy of ``annotated_files`` from which
    # we remove files after mapping them to annotation,
    # to validate that function worked,
    # by making sure there are no items left in this copy after the loop.
    # If there is 1:1 mapping then there should be no items left.
    annotated_annot_map = {}
    annotated_files_copy = copy.deepcopy(annotated_files)
    for annotated_file in annotated_files:
        # stem annotated file so we can find audio OR spect files
        # that match with stems from each annot.notated_path;
        # e.g. find '~/path/to/llb3/llb3_0003_2018_04_23_14_18_54.wav.mat' that
        # should match with ``Annotation(notated_path='llb3_0003_2018_04_23_14_18_54.wav')``
        annotated_file_stem = audio_filename_from_path(annotated_file)
        annot = audio_stem_annot_map[annotated_file_stem]
        annotated_annot_map[annotated_file] = annot
        annotated_files_copy.remove(annotated_file)

    if len(annotated_files_copy) > 0:
        raise MapUsingNotatedPathError(
            "Could not map the following source files to annotations: "
            f"{annotated_files_copy}"
        )
    return annotated_annot_map


class MapUsingExtensionError(BaseException):
    """Error raised when :func:`vak.annotation._map_using_ext`
    cannot map the filename of an annotation file to the name
    of an annotated file"""
    pass


def _map_using_ext(annotated_files: list[PathLike],
                   annot_list: list[crowsetta.Annotation],
                   annot_format: str,
                   method: str,
                   annotated_ext: str | None = None) -> dict:
    """Map a list of annotated files to a :class:`list` of
    :class:`crowsetta.Annotation` instances,
    by either removing the extension of the annotation format,
    or replacing it with the extension of the annotated file format.

    This function assumes a one-to-one mapping between
    annotation files and the files they annotate.

    and that the name of the annotated file is
    the name of the annotation file with its
    format-specific extension removed,
    e.g., a file in a csv-based format named 'bird1.wav.csv'
    annotates a file named `bird1.wav`.

    One of two helper functions used by
    :func:`~.vak.annotation.map_annotated_to_annot`.

    Parameters
    ----------
    annotated_files : list
        List of paths to the annotated files.
    annot_list : list
        List of ``crowsetta.Annotation`` instances.
    annot_format : str
        String name of annotation format
        Valid names are listed in
        ``vak.constants.VALID_ANNOT_FORMATS``.
    method: str
        The "method" used to determine the annotated
        file name from the annotation file name.
        One of {'remove', 'replace'}.
        Corresponds to either removing the extension
        for the annotation file format, or replacing
        its extension with the extension of the annotated
        format.

    Returns
    -------
    annotated_annot_map : dict
        Where each key is path to annotated file, and
        its value is the corresponding ``crowsetta.Annotation``.
    """
    if method not in {'remove', 'replace'}:
        raise ValueError(
            f"`method` must be one of: {{'remove', 'replace'}}, but was: '{method}'"
        )

    if method == 'replace' and annotated_ext is None:
        raise ValueError(
            "`method` was 'replace' but no `annotated_ext` was provided. "
            "Please use this argument to indicate the extension of the file format "
            "of the annotated files."
        )

    if annotated_ext:
        if not annotated_ext.startswith('.'):
            annotated_ext = f'.{annotated_ext}'

    annotated_files = [
        pathlib.Path(annotated_file) for annotated_file in annotated_files
    ]

    annot_class = crowsetta.formats.by_name(annot_format)

    # ---- make the dict that maps name of annotated files to crowsetta.Annotations
    annotated_annot_map = {}
    for annot in annot_list:
        annotated_name = None

        if isinstance(annot_class.ext, str):
            # NOTE that by convention the `ext` attribute
            # of all Crowsetta annotation format classes
            # begins with a period
            annotated_name = annot.annot_path.name.replace(annot_class.ext, '')
        elif isinstance(annot_class.ext, tuple):
            for ext in annot_class.ext:
                if annot_class.name.endswith(ext):
                    annotated_name = annot.annot_path.name.replace(ext, '')
                    break

        if annotated_name is None:
            raise MapUsingExtensionError(
                "Could not determine annotated file from annotation path, "
                f"using extension '{annot_class.ext}' from class '{annot_class.__name__}' "
                f"associated with format '{annot_format}'. "
                f"Annotation path was:\n{annot.annot_path}"
            )

        # NOTE we don't have to do anything else for method=='remove'
        # since we just removed the extension
        if method == 'replace':
            annotated_name = annotated_name + annotated_ext

        annotated_annot_map[annotated_name] = annot

    # Make a copy of ``annotated_files`` from which
    # we remove files after mapping them to annotation,
    # to validate that function worked,
    # by making sure there are no items left in this copy after the loop.
    # If there is 1:1 mapping then there should be no items left.
    annotated_files_copy = copy.deepcopy(annotated_files)
    for annotated_file in annotated_files:
        try:
            annot = annotated_annot_map[annotated_file.name]
            annotated_files_copy.remove(annotated_file)
        except KeyError as e:
            raise MapUsingExtensionError(
                f"Did not find an annotation that produced annotated file: {annotated_file}"
            ) from e

    if len(annotated_files_copy) > 0:
        raise MapUsingExtensionError(
            "Could not map the following source files to annotations: "
            f"{annotated_files_copy}"
        )
    return annotated_annot_map



def _map_replacing_ext(annotated_files: list[PathLike],
                       annot_list: list[crowsetta.Annotation],
                       annotated_ext: str) -> dict[crowsetta.Annotation: PathLike]:
    """Map a list of annotated files to a :class:`list` of
    :class:`crowsetta.Annotation` instances,
    by removing the extension of the annotation format.

    This function assumes a one-to-one mapping between
    annotation files and the files they annotate,
    and that the name of the annotated file is
    the name of the annotation file with its
    format-specific extension replaced
    with the extension of the annotated file,
    e.g., if a  file in a csv-based format named 'bird1.csv'
    annotates a wav file, the wav file would be named 'bird1.wav'.

    First, replaces the extension of each
    ``Annotation``s ``annot_path`` attribute
    with ``annotated_ext``, and then
    verifies that each path generated
    this way can be matched with a path in
    ``annotated_files``, without any
    missing or any extras.

    One of three helper functions used by
    :func:`~.vak.annotation.map_annotated_to_annot`.

    Parameters
    ----------
    annotated_files : list
        List of paths to the annotated files.
    annot_list : list
        List of ``crowsetta.Annotation`` instances.
    annotated_ext : str
        Extension of the annotated files.
        E.g., an audio format like 'wav',
        or an array file format like 'mat'
        or 'npz' that can contain spectrograms.

    Returns
    -------
    annotated_annot_map : dict
        Where each key is path to annotated file, and
        its value is the corresponding ``crowsetta.Annotation``.
    """
    # sanitize arg, so we are always working with
    # lowercase extension w/out period
    if not annotated_ext.startswith('.'):
        annotated_ext = f'.{annotated_ext}'

    # we make two lists that we will match:
    # one from annotations, the other from annotated files
    annot_filenames_with_annotated_ext = [
        annot.annot_path.stem + f'.{annotated_ext}'
        for annot in annot_list
    ]
    annotated_files_just_names = [
        # make actual annotated file extension lower too; ensures we can match
        pathlib.Path(annotated_file).stem + pathlib.Path(annotated_file).suffix.lower()
        for annotated_file in annotated_files
    ]
    # Make a copy of annotated files
    # from which we remove entries after mapping them to an annotation,
    # to validate that function worked,
    # by making sure there are no items left in this copy after the loop.
    # If there is 1:1 mapping then there should be no items left.
    annotated_files_copy = copy.deepcopy(annotated_files)
    annotated_annot_map = {}
    for annot, annot_filename_with_annotated_ext in zip(annot_list,
                                                    annot_filenames_with_annotated_ext):
        try:
            idx = annotated_files_just_names.index(
                annot_filename_with_annotated_ext
            )
        except ValueError as e:
            raise AnnotatedFilenameNotFound(
                f"Unable to match annotation with annotated file, "
                f"using the annotation file's path:\n{annot.annot_path}\n"
                f"The expected name of the annotated file, "
                f"using the ``annotated_ext`` '{annotated_ext}`, "
                f"was:\n{annot_filename_with_annotated_ext}"
            ) from e
        annotated_annot_map[annotated_files[idx]] = annot
        annotated_files_copy.remove(annotated_files[idx])

    if len(annotated_files_copy) > 0:
        raise ValueError(
            "Could not map the following source files to annotations: "
            f"{annotated_files_copy}"
        )
    return annotated_annot_map


def map_annotated_to_annot(annotated_files: Union[list, np.array],
                           annot_list: list[crowsetta.Annotation],
                           annot_format: str) -> dict:
    """Map annotated files,
    i.e. audio or spectrogram files,
    to their corresponding annotations.

    This function implements the three different ways that
    vak can map annotated files to their annotations.
    The first is when a single annotation file contains
    multiple annotations, and so the format by necessity
    must include the file annotated by each annotation.
    The second assumes that the annotated file can be determined
    programmatically by removing the extension from the annotation file,
    e.g. 'bird1.wav.csv' -> 'bird1.wav'.
    The third assumes that the annotated file can be determined
    by replacing the extension of the annotation file
    with the extension of the annotated file,
    e.g. 'bird1.csv' -> 'bird1.wav'.

    Returns a :class:`dict` where each key
    is a path to an annotated file,
    and the value for each key
    is a :class:`crowsetta.Annotation`.

    Mapping is done with three helper functions,
    :func:`~vak.annotation._map_using_notated_path`,
    :func:`~vak.annotation._map_removing_ext`, and
    :func:`~vak.annotation._map_replacing_ext`.

    The function :func:`~vak.annotation._map_using_notated_path`
    is used for annotation formats that include
    the name of the annotated file.
    The names of these formats (in :module:`crowsetta`) are:
    {'birdsong-recognition-dataset', 'generic-seq', 'yarden'}.

    The other two functions are used for all other formats,
    and they assume a one-to-one mapping from annotation file
    to annotated file.
    The first, :func:`~vak.annotation._map_removing_ext`,
    assumes that the name of the annotated file
    can be found by removing the extension of the annotation
    format, e.g., 'bird1.wav.csv` -> 'bird1.wav'.
    The second, that is used if the first fails,
    assumes the name of the annotated file
    can be found by replacing the extension of the annotation
    format with the extension of the annotated files.

    Parameters
    ----------
    annotated_files : list
        Of paths to audio or spectrogram files.
    annot_list : list
        Of Annotations corresponding to files in annotated_files
    audio_ext : str
        Extension of audio files.
        Default is None, in which case this function will
        look for extensions of any valid audio format
        (listed as ``vak.constants.VALID_AUDIO_FORMAT``).
        Specifying the format provides a slight speed up.
    annotated_ext : str
        Extension of the annotated files.
        Default is None

    Notes
    -----

    For more detail, please see
    the page on file naming conventions in the
    reference section of the documentation:
    https://vak.readthedocs.io/en/latest/reference/filenames.html
    """
    if type(annotated_files) == np.ndarray:  # e.g., vak DataFrame['spect_path'].values
        annotated_files = annotated_files.tolist()

    if annot_format in ('birdsong-recognition-dataset', 'yarden', 'generic-seq'):
        annotated_annot_map = _map_using_notated_path(annotated_files, annot_list)
    else:
        try:
            annotated_annot_map = _map_removing_ext(annotated_files, annot_list, annot_format)
        except AudioFilenameNotFound:
            try:
                annotated_annot_map = _map_replacing_ext(annotated_files, annot_list, annot_format)
            except AnnotatedFilenameNotFound as e:
                raise ValueError(
                    'Could not map annotated files to annotations.\n'
                    'Please see this section in the `vak` documentation:\n'
                    'https://vak.readthedocs.io/en/latest/howto/howto_prep_annotate.html#how-does-vak-know-which-annotations-go-with-which-annotated-files'
                ) from e

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
    if duration <= 0:
        raise ValueError(
            f"Duration less than or equal to zero passed to `has_unlabeled`.\n"
            f"Value for `duration`: {duration}.\nValue for `annot`: {annot}"
        )
    if duration > 0 and len(annot.seq.segments) < 1:
        # Handle edge case where there are no annotated segments in annotation file
        # See https://github.com/vocalpy/vak/issues/378
        return True
    has_unlabeled_intervals = np.any((annot.seq.onsets_s[1:] - annot.seq.offsets_s[:-1]) > 0.)
    has_unlabeled_before_first_onset = annot.seq.onsets_s[0] > 0.
    has_unlabeled_after_last_offset = duration - annot.seq.offsets_s[-1] > 0.
    return has_unlabeled_intervals or has_unlabeled_before_first_onset or has_unlabeled_after_last_offset
