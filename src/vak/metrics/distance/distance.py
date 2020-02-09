from . import functional as F

__all__ = [
    'Levenshtein', 'SegmentErrorRate'
]


class Levenshtein:
    """levenshtein distance
    returns number of deletions, insertions, or substitutions
    required to convert source string into target string.

    Parameters
    ----------
    source, target : str

    Returns
    -------
    distance : int
        number of deletions, insertions, or substitutions
        required to convert source into target.
    """
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        return F.levenshtein(y_pred, y_true)


class SegmentErrorRate:
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
    Levenshtein distance(y_pred, y_true) / len(y_true)
    """
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        return F.segment_error_rate(y_pred, y_true)