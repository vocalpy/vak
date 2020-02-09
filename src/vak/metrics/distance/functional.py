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

    from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


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
        raise TypeError('Both `y_true` and `y_pred` must be of type `str')

    return levenshtein(y_pred, y_true) / len(y_true)
