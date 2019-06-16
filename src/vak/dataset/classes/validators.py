import numpy as np


def asarray_if_not(val):
    if val is None:
        return None
    else:
        if type(val) == np.ndarray:
            return val
        else:
            return np.asarray(val)