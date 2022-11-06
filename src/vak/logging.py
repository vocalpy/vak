"""utility functions for logging"""
import logging
from pathlib import Path
import sys
import warnings

from . import timenow
from .__about__ import __version__


logger = logging.getLogger('vak')  # 'base' logger


def config_logging_for_cli(log_dst: str,
                           log_stem: str,
                           level='info',
                           timestamp=None,
                           force=False):
    """Configure logging for a run of the cli.

    Called by `vak.cli` functions. Allows logging
    to also be configured differently by a user
    that interacts directly with the library through e.g. `vak.core`.

    Parameters
    ----------
    log_dst : str, Path
        Destination directory where log file should be saved
    log_stem : str
        The "stem" of the filename for the log file.
        A timestamp is added to this stem,
        so that the final filename is ``f"{caller}_{timestamp}.log"``.
        Usually set to the name of the function that called this one,
        e.g. 'train' or 'prep'.
    level : str
        Logging level, e.g. "INFO". Passed in to ``logger.setLevel``.
    timestamp : str
        Time stamp, included in log file name.
        If None, defaults to `datetime.now().strftime('%y%m%d_%H%M%S')`.
    force : bool
        If True, forces this function to remove the old handlers
        and add new handlers. If False, this function will raise an
        error when the logger already has handlers,
        to avoid silent failures.
    """
    log_dst = Path(log_dst)
    if not log_dst.is_dir():
        raise NotADirectoryError(
            f"destination for log file is not a directory: {log_dst}"
        )

    if timestamp is None:
        timestamp = timenow.get_timenow_as_str()
    logfile_name = log_dst.joinpath(f"{log_stem}_{timestamp}.log")

    if logger.hasHandlers():
        if force:
            logger.handlers.clear()
        else:
            warnings.warn(
                f"Logger already has handlers attached:\n{logger.handlers}."
                f"Will not add new handlers, to avoid duplicate messages "
                f"and corrupted logs. To override, set ``force=True`` "
                f"when calling this function."
            )
            return

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfile_name)
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)


def log_version(logger: logging.Logger) -> None:
    logger.info(
            f"vak version: {__version__}"
    )
