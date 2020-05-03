"""constants used by multiple modules.
Defined here to avoid circular imports.
"""
from functools import partial

import crowsetta
import numpy as np
from evfuncs import load_cbin
from scipy.io import wavfile, loadmat

# ---- audio files ----
def swap_return_tuple_elements(func):
    def new_f(*args, **kwargs):
        return_tuple = func(*args, **kwargs)
        return return_tuple[1], return_tuple[0]
    return new_f


load_cbin = swap_return_tuple_elements(load_cbin)
AUDIO_FORMAT_FUNC_MAP = {
    'cbin': load_cbin,
    'wav': wavfile.read
}

VALID_AUDIO_FORMATS = list(AUDIO_FORMAT_FUNC_MAP.keys())

# ---- spectrogram files ----
SPECT_FORMAT_LOAD_FUNCTION_MAP = {
    'mat': partial(loadmat, squeeze_me=True),
    'npz': np.load,
}
VALID_SPECT_FORMATS = list(SPECT_FORMAT_LOAD_FUNCTION_MAP.keys())

# ---- annotation files ----
VALID_ANNOT_FORMATS = crowsetta.formats._INSTALLED
NO_ANNOTATION_FORMAT = 'none'
