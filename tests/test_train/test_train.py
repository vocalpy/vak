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
        ("wav", None, "birdsong-recognition-dataset", "TweetyNet",
         'vak.train.train_.train_frame_classification_model'),
        (None, "mat", "yarden", "TweetyNet",
         'vak.train.train_.train_frame_classification_model'),
        ("cbin", None, "notmat", "ConvEncoderUMAP",
         'vak.train.train_.train_parametric_umap_model'),
    ],
)
def test_train(
    audio_format, spect_format, annot_format, model_name, train_function_to_mock,
    specific_config, tmp_path
):
    """Test that :func:`vak.train.train` dispatches to the correct model-specific
    training functions"""
    root_results_dir = tmp_path.joinpath("test_train_root_results_dir")
    root_results_dir.mkdir()

    options_to_change = [
        {
            "section": "TRAIN",
            "option": "root_results_dir",
            "value": str(root_results_dir),
        },
        {"section": "TRAIN", "option": "device", "value": 'cpu'},
    ]

    toml_path = specific_config(
        config_type="train",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.train.model)

    results_path = tmp_path / 'results_path'
    results_path.mkdir()

    with mock.patch(train_function_to_mock, autospec=True) as mock_train_function:
        vak.train.train(
            model_name=model_name,
            model_config=model_config,
            dataset_path=cfg.train.dataset_path,
            batch_size=cfg.train.batch_size,
            num_epochs=cfg.train.num_epochs,
            num_workers=cfg.train.num_workers,
            train_transform_params=cfg.train.train_transform_params,
            train_dataset_params=cfg.train.train_dataset_params,
            val_transform_params=cfg.train.val_transform_params,
            val_dataset_params=cfg.train.val_dataset_params,
            checkpoint_path=cfg.train.checkpoint_path,
            spect_scaler_path=cfg.train.spect_scaler_path,
            results_path=results_path,
            normalize_spectrograms=cfg.train.normalize_spectrograms,
            shuffle=cfg.train.shuffle,
            val_step=cfg.train.val_step,
            ckpt_step=cfg.train.ckpt_step,
            patience=cfg.train.patience,
            device=cfg.train.device,
        )
        assert mock_train_function.called
