"""utility functions for logging"""
import datetime
import logging
from pathlib import Path
import sys


def get_logger(log_dst,
               caller,
               logger_name,
               timestamp=None,
               level='INFO'):
    """get a logger

    Parameters
    ----------
    log_dst : str, Path
        destination directory where log file should be saved
    caller : str
        function that called get_logger, e.g. 'train' or 'prep'.
        Included in log file name.
    timestamp : str
        time stamp, included in log file name.
        If None, defaults to `datetime.now().strftime('%y%m%d_%H%M%S')`.
    logger_name : str
        name to give logger when instantiating.
    level : str
        logging level. Default is 'INFO'.

    Returns
    -------
    logger : logging.Logger
        as returned by logging.getLogger.
        Will save logs to file as well as log to stdout.
    """
    log_dst = Path(log_dst)
    if not log_dst.is_dir():
        raise NotADirectoryError(
            f'destination for log file is not a directory: {log_dst}'
        )
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if timestamp is None:
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    logfile_name = log_dst.joinpath(
        f'{caller}_{timestamp}.log'
    )
    logger.addHandler(logging.FileHandler(logfile_name))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
