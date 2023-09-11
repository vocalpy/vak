"""Tests for vak.predict.predict module."""
from unittest import mock
import pytest

import vak.cli.predict
import vak.config
import vak.common.constants
import vak.common.paths


@pytest.mark.parametrize(
    "audio_format, spect_format, annot_format, model_name, predict_function_to_mock",
    [
        ("cbin", None, "notmat", "TweetyNet",
         'vak.predict.predict_.predict_with_frame_classification_model'),
    ],
)
def test_predict(
    audio_format, spect_format, annot_format, model_name, predict_function_to_mock,
    specific_config, tmp_path
):
    """Test that :func:`vak.predict.predict` dispatches to the correct model-specific
    training functions"""
    output_dir = tmp_path.joinpath(
        f"test_predict_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "PREDICT", "option": "output_dir", "value": str(output_dir)},
        {"section": "PREDICT", "option": "device", "value": 'cpu'},
    ]

    toml_path = specific_config(
        config_type="predict",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.predict.model)

    results_path = tmp_path / 'results_path'
    results_path.mkdir()

    with mock.patch(predict_function_to_mock, autospec=True) as mock_predict_function:
        vak.predict.predict(
            model_name=model_name,
            model_config=model_config,
            dataset_path=cfg.predict.dataset_path,
            checkpoint_path=cfg.predict.checkpoint_path,
            labelmap_path=cfg.predict.labelmap_path,
            num_workers=cfg.predict.num_workers,
            transform_params=cfg.predict.transform_params,
            dataset_params=cfg.predict.dataset_params,
            timebins_key=cfg.spect_params.timebins_key,
            spect_scaler_path=cfg.predict.spect_scaler_path,
            device=cfg.predict.device,
            annot_csv_filename=cfg.predict.annot_csv_filename,
            output_dir=cfg.predict.output_dir,
            min_segment_dur=cfg.predict.min_segment_dur,
            majority_vote=cfg.predict.majority_vote,
            save_net_outputs=cfg.predict.save_net_outputs,
        )
        assert mock_predict_function.called
