import os
from glob import glob


def _files_from_dir(dir_path, ext):
    """helper function that gets all files with a given extension
    from a directory or its sub-directories.

    If no files with the specified extension are found in the directory, then
    the function recurses into all sub-directories and returns any files with
    the extension in those sub-directories.

    Parameters
    ----------
    dir_path : str
        path to target directory
    ext : str
        file extension to search for

    Returns
    -------
    files : list
        of paths to files with specified file extension

    Notes
    -----
    used by vak.dataset.audio.files_from_dir and vak.dataset.annot.files_from_dir
    """
    wildcard_with_extension = f'*.{ext}'
    files = glob(os.path.join(dir_path, wildcard_with_extension))
    if len(files) == 0:
        # if we don't any files with extension, look in sub-directories
        files = []
        subdirs = glob(os.path.join(dir_path, '*/'))
        for subdir in subdirs:
            files.extend(
                glob(os.path.join(dir_path, subdir, wildcard_with_extension))
            )

    if len(files) == 0:
        raise FileNotFoundError(
            f'No files with extension {ext} found in '
            f'{dir_path} or immediate sub-directories'
        )

    return files
