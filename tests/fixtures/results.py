"""Fixtures made with results from generated test data,
in ./tests/data_for_tests/generated/results"""
from collections import defaultdict

from .model import MODELS
from .test_data import GENERATED_RESULTS_DATA_ROOT


GENERATED_LEARNCURVE_RESULTS_ROOT = GENERATED_RESULTS_DATA_ROOT / 'learncurve'

SOURCE_ANNOT_PAIRS = [
    'audio_cbin_annot_notmat'
]

GENERATED_LEARNCURVE_RESULTS_BY_MODEL = defaultdict(list)

for source_annot_pair in SOURCE_ANNOT_PAIRS:
    source_annot_root = GENERATED_LEARNCURVE_RESULTS_ROOT / source_annot_pair
    for model in MODELS:
        model_root = source_annot_root / model
        results_paths = sorted(model_root.glob('results_*'))
        GENERATED_LEARNCURVE_RESULTS_BY_MODEL[model].extend(results_paths)
