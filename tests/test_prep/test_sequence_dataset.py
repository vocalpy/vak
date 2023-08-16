import pandas as pd
import pytest

import vak


@pytest.mark.parametrize(
    'config_type, audio_format, spect_format, annot_format, expected_result',
    [
        ("train", "cbin", None, "notmat", True),
        ("train", "wav", None, "birdsong-recognition-dataset", True),
        ("train", None, "mat", "yarden", True),
    ]
)
def test_has_unlabeled_segments(config_type,
                                audio_format,
                                spect_format,
                                annot_format,
                                expected_result,
                                model,
                                specific_config_toml,
                                specific_dataset_csv_path):
    dataset_csv_path = specific_dataset_csv_path(config_type,
                                                 model,
                                                 annot_format,
                                                 audio_format,
                                                 spect_format)

    dataset_df = pd.read_csv(dataset_csv_path)
    has_unlabeled = vak.prep.sequence_dataset.has_unlabeled_segments(dataset_df)
    assert has_unlabeled == expected_result
