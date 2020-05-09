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


def range_str(range_str, sort=True):
    """Generate range of ints from a formatted string,
    then convert range from int to str

    Examples
    --------
    >>> range_str('1-4,6,9-11')
    ['1','2','3','4','6','9','10','11']

    Takes a range in form of "a-b" and returns
    a list of numbers between a and b inclusive.
    Also accepts comma separated ranges like "a-b,c-d,f"  which will
    return a list with numbers from a to b, c to d, and f.

    Parameters
    ----------
    range_str : str
        of form 'a-b,c', where a hyphen indicates a range
        and a comma separates ranges or single numbers
    sort : bool
        If True, sort output before returning. Default is True.

    Returns
    -------
    list_range : list
        of integer values converted to single-character strings, produced by parsing range_str
    """
    # adapted from
    # http://code.activestate.com/recipes/577279-generate-list-of-numbers-from-hyphenated-and-comma/
    s = "".join(range_str.split())  # removes white space
    list_range = []
    for substr in range_str.split(','):
        subrange = substr.split('-')
        if len(subrange) not in [1, 2]:
            raise SyntaxError("unable to parse range {} in labelset {}."
                              .format(subrange, substr))
        list_range.extend(
            [int(subrange[0])]
        ) if len(subrange) == 1 else list_range.extend(
            range(int(subrange[0]), int(subrange[1]) + 1))

    if sort:
        list_range.sort()

    return [str(list_int) for list_int in list_range]


def labelset_from_toml_value(value):
    """convert value for 'labelset' option from .toml config file into a set

    Parameters
    ----------
    value : str, list
        value assigned to 'labelset' option in a .toml file

    Returns
    -------
    labelset : set
        of single-character strings.

    Notes
    -----
    If value is a str, and it starts with "range:", then everything after range is converted to
    some range of integers, by passing the string to vak.config.converters.range_str,
    and the returned list is converted to a set. E.g. "range: 1-5" becomes {'1', '2', '3', '4', '5'}.
    Other strings that do not start with "range:" are just converted to a set. E.g. "abc" becomes {'a', 'b', 'c'}.

    If value is a list, returns set(value). If all values in list are int,
    they will be converted to strings, e.g. [1, 2, 3, 4, 5] becomes {'1', '2', '3', '4', '5'}.
    If all values are string, any that begin with "range:" will be passed to vak.config.converters.range_str.

    Examples
    --------
    >>> labelset_from_toml_value('1235')
    {'1', '2', '3', '5'}
    >>> labelset_from_toml_value('range: 1-3, 5')
    {'1', '2', '3', '5'}
    >>> labelset_from_toml_value([1, 2, 3])
    {'1', '2', '3'}
    >>> labelset_from_toml_value(['a', 'b', 'c'])
    {'a', 'b', 'c'}
    >>> labelset_from_toml_value(['range: 1-3', 'a-'])
    {'-', '1', '2', '3', 'a'}
    """
    if type(value) is str:
        if value.startswith('range:'):
            value = value.replace('range:', '')
            return set(range_str(value))
        else:
            return set(value)
    elif type(value) is list:
        if all([type(el) == int for el in value]):
            return set([str(el) for el in value])
        elif all([type(el) == str for el in value]):
            labelset = []
            for el in value:
                if el.startswith('range:'):
                    el = el.replace('range:', '')
                    labelset.extend(range_str(el))
                else:
                    labelset.extend(
                        list(el)  # in case element is a string with multiple characters
                    )
            return set(labelset)
