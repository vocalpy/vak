import numpy as np


def levenshtein(source, target):
    """Levenshtein distance: number of deletions, insertions,
    or substitutions required to convert source string
    into target string.

    Parameters
    ----------
    source, target : str

    Returns
    -------
    distance : int
        number of deletions, insertions, or substitutions
        required to convert source into target.

    adapted from https://github.com/toastdriven/pylev/blob/master/pylev.py
    to fix issues with the Numpy implementation in
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if source == target:
        return 0

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    len_source = source.size
    len_target = target.size

    if len_source == 0:
        return len_target
    if len_target == 0:
        return len_source

    if len_source > len_target:
        source, target = target, source
        len_source, len_target = len_target, len_source

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    d0 = np.arange(len_target + 1)
    d1 = np.arange(len_target + 1)
    for i in range(len_source):
        d1[0] = i + 1
        for j in range(len_target):
            cost = d0[j]

            if source[i] != target[j]:
                cost += 1  # substitution

                x_cost = d1[j] + 1  # insertion
                if x_cost < cost:
                    cost = x_cost

                y_cost = d0[j + 1] + 1
                if y_cost < cost:
                    cost = y_cost

            d1[j + 1] = cost

        d0, d1 = d1, d0

    return d0[-1]


def segment_error_rate(y_pred, y_true):
    """Levenshtein edit distance normalized by length of true sequence.
    Also known as word error distance; here applied to other vocalizations
    in addition to speech.

    Parameters
    ----------
    y_pred : str
        predicted labels for a series of songbird syllables
    y_true : str
        ground truth labels for a series of songbird syllables

    Returns
    -------
    Levenshtein distance / len(y_true)
    """
    if type(y_true) != str or type(y_pred) != str:
        raise TypeError("Both `y_true` and `y_pred` must be of type `str")

    # handle divide by zero edge cases
    if len(y_true) == 0 and len(y_pred) == 0:
        return 0.
    elif len(y_true) == 0 and len(y_pred) != 0:
        raise ValueError(
            f'segment error rate is undefined when length of y_true is zero'
        )

    return levenshtein(y_pred, y_true) / len(y_true)
