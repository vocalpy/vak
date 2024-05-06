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
        specific_config_toml_path, tmp_path
):
    """Test that :func:`vak.eval.eval` dispatches to the correct model-specific
    training functions"""
    output_dir = tmp_path.joinpath(
        f"test_eval_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    keys_to_change = [
        {"table": "eval", "key": "output_dir", "value": str(output_dir)},
        {"table": "eval", "key": "trainer", "value": {"accelerator": "cpu", "devices": 1}},
    ]

    toml_path = specific_config_toml_path(
        config_type="eval",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)

    results_path = tmp_path / 'results_path'
    results_path.mkdir()

    with mock.patch(eval_function_to_mock, autospec=True) as mock_eval_function:
        vak.eval.eval(
            model_config=cfg.eval.model.asdict(),
            dataset_config=cfg.eval.dataset.asdict(),
            trainer_config=cfg.eval.trainer.asdict(),
            checkpoint_path=cfg.eval.checkpoint_path,
            labelmap_path=cfg.eval.labelmap_path,
            output_dir=cfg.eval.output_dir,
            num_workers=cfg.eval.num_workers,
            batch_size=cfg.eval.batch_size,
            frames_standardizer_path=cfg.eval.frames_standardizer_path,
            post_tfm_kwargs=cfg.eval.post_tfm_kwargs,
        )

        assert mock_eval_function.called
