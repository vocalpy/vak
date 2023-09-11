import pathlib
from unittest import mock

import pandas as pd
import pytest

import vak


@pytest.mark.parametrize(
    "config_type, audio_format, spect_format, annot_format, dataset_prep_func_to_mock",
    [
        ("eval", "cbin", None, "notmat", "vak.prep.prep_.prep_frame_classification_dataset"),
        ("learncurve", "cbin", None, "notmat", "vak.prep.prep_.prep_frame_classification_dataset"),
        ("predict", "cbin", None, "notmat", "vak.prep.prep_.prep_frame_classification_dataset"),
        ("predict", "wav", None, "birdsong-recognition-dataset", "vak.prep.prep_.prep_frame_classification_dataset"),
        ("train", "cbin", None, "notmat", "vak.prep.prep_.prep_frame_classification_dataset"),
        ("train", "wav", None, "birdsong-recognition-dataset", "vak.prep.prep_.prep_frame_classification_dataset"),
        ("train", None, "mat", "yarden", "vak.prep.prep_.prep_frame_classification_dataset"),
    ],
)
def test_prep(
    config_type,
    audio_format,
    spect_format,
    annot_format,
    dataset_prep_func_to_mock,
    specific_config,
    default_model,
    tmp_path,
):
    # ---- set up
    output_dir = tmp_path.joinpath(
        f"test_prep_{config_type}_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = [
        {
            "section": "PREP",
            "option": "output_dir",
            "value": str(output_dir),
        },
    ]
    toml_path = specific_config(
        config_type=config_type,
        model=default_model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    purpose = config_type.lower()

    # ---- test
    with mock.patch(dataset_prep_func_to_mock, autospec=True) as mocked_dataset_prep_func:
        mocked_dataset_prep_func.return_value = (pd.DataFrame.from_records([]), pathlib.Path('./fake/dataset/path'))
        _ = vak.prep.prep(
            data_dir=cfg.prep.data_dir,
            dataset_type=cfg.prep.dataset_type,
            input_type=cfg.prep.input_type,
            purpose=purpose,
            audio_format=cfg.prep.audio_format,
            spect_format=cfg.prep.spect_format,
            spect_params=cfg.spect_params,
            annot_format=cfg.prep.annot_format,
            annot_file=cfg.prep.annot_file,
            labelset=cfg.prep.labelset,
            output_dir=cfg.prep.output_dir,
            train_dur=cfg.prep.train_dur,
            val_dur=cfg.prep.val_dur,
            test_dur=cfg.prep.test_dur,
            train_set_durs=cfg.prep.train_set_durs,
            num_replicates=cfg.prep.num_replicates,
        )

        assert mocked_dataset_prep_func.called
