"""Tests for vak.train.train function."""
from unittest import mock

import pytest

import vak.config
import vak.common.constants
import vak.common.paths
import vak.train


@pytest.mark.parametrize(
    "audio_format, spect_format, annot_format, model_name, train_function_to_mock",
    [
        ("cbin", None, "notmat", "TweetyNet",
         'vak.train.train_.train_frame_classification_model'),
        (None, "mat", "yarden", "TweetyNet",
         'vak.train.train_.train_frame_classification_model'),
        ("cbin", None, "notmat", "ConvEncoderUMAP",
         'vak.train.train_.train_parametric_umap_model'),
    ],
)
def test_train(
    audio_format, spect_format, annot_format, model_name, train_function_to_mock,
        specific_config_toml_path, tmp_path
):
    """Test that :func:`vak.train.train` dispatches to the correct model-specific
    training functions"""
    root_results_dir = tmp_path.joinpath("test_train_root_results_dir")
    root_results_dir.mkdir()

    keys_to_change = [
        {
            "table": "train",
            "key": "root_results_dir",
            "value": str(root_results_dir),
        },
        {"table": "train", "key": "trainer", "value": {"accelerator": "cpu", "devices": 1}},
    ]

    toml_path = specific_config_toml_path(
        config_type="train",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)

    results_path = tmp_path / 'results_path'
    results_path.mkdir()

    with mock.patch(train_function_to_mock, autospec=True) as mock_train_function:
        vak.train.train(
            model_config=cfg.train.model.asdict(),
            dataset_config=cfg.train.dataset.asdict(),
            trainer_config=cfg.train.trainer.asdict(),
            batch_size=cfg.train.batch_size,
            num_epochs=cfg.train.num_epochs,
            num_workers=cfg.train.num_workers,
            checkpoint_path=cfg.train.checkpoint_path,
            frames_standardizer_path=cfg.train.frames_standardizer_path,
            results_path=results_path,
            standardize_frames=cfg.train.standardize_frames,
            shuffle=cfg.train.shuffle,
            val_step=cfg.train.val_step,
            ckpt_step=cfg.train.ckpt_step,
            patience=cfg.train.patience,
        )
        assert mock_train_function.called
