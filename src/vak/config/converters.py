from pathlib import Path
from distutils.util import strtobool

from ..util.general import range_str


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


def labelset_from_toml_value(value):
    """convert value for 'labelset' option from config file into a set

    Parameters
    ----------
    value : str, list
        if value is a list, returns set(value).
        If value is a str, and it starts with "range:", then everything after range is passed to
        vak.util.general.range_str, and the returned list is converted to a set.
        Other strings that do not start with "range:" are just converted to a set.

    Returns
    -------
    labelset : set

    Examples
    --------
    >>> labelset_from_toml_value([1, 2, 3])
    {1, 2, 3}
    >>> labelset_from_toml_value('range: 1-3, 5')
    {'1', '2', '3', '5'}
    >>> labelset_from_toml_value('1235')
    {'1', '2', '3', '5'}
    """
    if type(value) is list:
        return set(value)
    elif type(value) is str:
        if value.startswith('range:'):
            value = value.replace('range:', '')
            return set(range_str(value))
        else:
            return set(value)
