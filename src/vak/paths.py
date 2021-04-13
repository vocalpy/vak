"""functions for working with paths"""
from pathlib import Path

from . import constants, timenow


def generate_results_dir_name_as_path(root_results_dir=None):
    """generates a name for a new results directory,
    returns as a path

    Parameters
    ----------
    root_results_dir : str, pathlib.Path
        root directory within which a new results directory will be made.
        Default is None, in which case the new results directory name
        will be relative to the current working directory.

    Returns
    -------
    results_path : pathlib.Path
        path to a new results directory, with name of the following format:
        ``f'{vak.constants.RESULTS_DIR_PREFIX}{vak.timenow.get_timenow_as_str}'``.
        e.g., ``results_210211_142329``.
        This function simply builds the name and path with a consistent format.
        To actually make this directory, call ``results_path.mkdir()``
    """
    if root_results_dir:
        root_results_dir = Path(root_results_dir)
    else:
        root_results_dir = Path(".")
    if not root_results_dir.is_dir():
        raise NotADirectoryError(
            f"root_results_dir not recognized as a directory: {root_results_dir}"
        )

    results_dirname = f"{constants.RESULTS_DIR_PREFIX}{timenow.get_timenow_as_str()}"
    return root_results_dir.joinpath(results_dirname)
