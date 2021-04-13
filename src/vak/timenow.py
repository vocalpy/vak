from datetime import datetime

from .constants import STRFTIME_TIMESTAMP


def get_timenow_as_str():
    f"""returns current time as a string,
    with the format specified by ``vak.constants.STRFTIME_TIMESTAMP``"""
    return datetime.now().strftime(STRFTIME_TIMESTAMP)
