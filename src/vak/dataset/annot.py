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

