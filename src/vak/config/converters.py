from pathlib import Path
from distutils.util import strtobool


def bool_from_str(value):
    if type(value) == bool:
        return value
    elif type(value) == str:
        return bool(strtobool(value))


def comma_separated_list(value):
    if type(value) is list:
        return value
    elif type(value) is str:
        return [element.strip() for element in value.split()]
    else:
        raise TypeError(
            f'unexpected type when converting to comma-separated list: {type(value)}'
        )


def expanded_user_path(value):
    return Path(value).expanduser()

