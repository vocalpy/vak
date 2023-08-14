"""constants used by multiple modules.
Defined here to avoid circular imports.
"""
from functools import partial

import crowsetta
import numpy as np
import soundfile
from evfuncs import load_cbin
from scipy.io import loadmat

# ---- audio files ----
AUDIO_FORMAT_FUNC_MAP = {
    "cbin": load_cbin,
    "flac": soundfile.read,
    "ogg": soundfile.read,
    "wav": soundfile.read,
}

VALID_AUDIO_FORMATS = list(AUDIO_FORMAT_FUNC_MAP.keys())

# ---- spectrogram files ----
SPECT_FORMAT_LOAD_FUNCTION_MAP = {
    "mat": partial(loadmat, squeeze_me=True),
    "npz": np.load,
}
VALID_SPECT_FORMATS = list(SPECT_FORMAT_LOAD_FUNCTION_MAP.keys())

# ---- valid types of training data, the $x$ that goes into a network
VALID_X_SOURCES = {"audio", "spect"}

# ---- annotation files ----
VALID_ANNOT_FORMATS = crowsetta.formats.as_list()
NO_ANNOTATION_FORMAT = "none"

# format for timestamps
STRFTIME_TIMESTAMP = "%y%m%d_%H%M%S"

# ---- results, from train / learncurve ----
RESULTS_DIR_PREFIX = "results_"

# ---- output (default) file extensions. Using the `pathlib` name "suffix" ----
ANNOT_CSV_SUFFIX = ".annot.csv"
NET_OUTPUT_SUFFIX = ".output.npz"
