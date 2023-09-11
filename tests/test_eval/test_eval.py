"""Tests for vak.eval.eval function."""
from unittest import mock

import pytest

import vak.config
import vak.common.constants
import vak.common.paths
import vak.eval


@pytest.mark.parametrize(
    "audio_format, spect_format, annot_format, model_name, eval_function_to_mock",
    [
        ("cbin", None, "notmat", "TweetyNet",
         'vak.eval.eval_.eval_frame_classification_model'),
        ("cbin", None, "notmat", "ConvEncoderUMAP",
         'vak.eval.eval_.eval_parametric_umap_model'),
    ],
)
def test_eval(
        audio_format, spect_format, annot_format, model_name, eval_function_to_mock,
        specific_config, tmp_path
):
    """Test that :func:`vak.eval.eval` dispatches to the correct model-specific
    training functions"""
    output_dir = tmp_path.joinpath(
        f"test_eval_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "EVAL", "option": "output_dir", "value": str(output_dir)},
        {"section": "EVAL", "option": "device", "value": 'cpu'},
    ]

    toml_path = specific_config(
        config_type="eval",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.eval.model)

    results_path = tmp_path / 'results_path'
    results_path.mkdir()

    with mock.patch(eval_function_to_mock, autospec=True) as mock_eval_function:
        vak.eval.eval(
            model_name=model_name,
            model_config=model_config,
            dataset_path=cfg.eval.dataset_path,
            checkpoint_path=cfg.eval.checkpoint_path,
            labelmap_path=cfg.eval.labelmap_path,
            output_dir=cfg.eval.output_dir,
            num_workers=cfg.eval.num_workers,
            batch_size=cfg.eval.batch_size,
            transform_params=cfg.eval.transform_params,
            dataset_params=cfg.eval.dataset_params,
            spect_scaler_path=cfg.eval.spect_scaler_path,
            device=cfg.eval.device,
            post_tfm_kwargs=cfg.eval.post_tfm_kwargs,
        )

        assert mock_eval_function.called
