"""constants used by multiple modules.
Defined here to avoid circular imports.
"""
from functools import partial

import crowsetta
import numpy as np
from evfuncs import load_cbin
from scipy.io import loadmat
import soundfile


AUDIO_FORMAT_FUNC_MAP = {
    'cbin': load_cbin,
    'wav': soundfile.read
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
