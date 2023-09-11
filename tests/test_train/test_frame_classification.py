"""Tests for vak.train.frame_classification module"""
import pathlib

import pytest

import vak.config
import vak.common.constants
import vak.common.paths
import vak.train


def assert_train_output_matches_expected(cfg: vak.config.config.Config, model_name: str,
                                         results_path: pathlib.Path):
    assert results_path.joinpath("labelmap.json").exists()

    if cfg.train.normalize_spectrograms or cfg.train.spect_scaler_path:
        assert results_path.joinpath("StandardizeSpect").exists()
    else:
        assert not results_path.joinpath("StandardizeSpect").exists()

    model_path = results_path.joinpath(model_name)
    assert model_path.exists()

    tensorboard_log = sorted(
        model_path.glob(f"lightning_logs/**/*events*")
    )
    assert len(tensorboard_log) == 1

    checkpoints_path = model_path.joinpath("checkpoints")
    assert checkpoints_path.exists()
    assert checkpoints_path.joinpath("checkpoint.pt").exists()
    if cfg.train.val_step is not None:
        assert checkpoints_path.joinpath("max-val-acc-checkpoint.pt").exists()


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name, audio_format, spect_format, annot_format",
    [
        ("TweetyNet", "cbin", None, "notmat"),
        ("TweetyNet", "wav", None, "birdsong-recognition-dataset"),
        ("TweetyNet", None, "mat", "yarden"),
    ],
)
def test_train_frame_classification_model(
    model_name, audio_format, spect_format, annot_format, specific_config, tmp_path, device
):
    results_path = vak.common.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()
    options_to_change = [
        {"section": "TRAIN", "option": "device", "value": device},
        {"section": "TRAIN", "option": "root_results_dir", "value": results_path}
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

    vak.train.frame_classification.train_frame_classification_model(
        model_name=cfg.train.model,
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

    assert_train_output_matches_expected(cfg, cfg.train.model, results_path)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name, audio_format, spect_format, annot_format",
    [
        ("TweetyNet", "cbin", None, "notmat"),
        ("TweetyNet", "wav", None, "birdsong-recognition-dataset"),
        ("TweetyNet", None, "mat", "yarden"),
    ],
)
def test_continue_training(
    model_name, audio_format, spect_format, annot_format, specific_config, tmp_path, device
):
    results_path = vak.common.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()
    options_to_change = [
        {"section": "TRAIN", "option": "device", "value": device},
        {"section": "TRAIN", "option": "root_results_dir", "value": results_path}
    ]
    toml_path = specific_config(
        config_type="train_continue",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.train.model)

    vak.train.frame_classification.train_frame_classification_model(
        model_name=cfg.train.model,
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

    assert_train_output_matches_expected(cfg, cfg.train.model, results_path)


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"section": "TRAIN", "option": "checkpoint_path", "value": '/obviously/doesnt/exist/ckpt.pt'},
        {"section": "TRAIN", "option": "spect_scaler_path", "value": '/obviously/doesnt/exist/SpectScaler'},
    ]
)
def test_train_raises_file_not_found(
    path_option_to_change, specific_config, tmp_path, device
):
    """Test that pre-conditions in `vak.train` raise FileNotFoundError
    when one of the following does not exist:
    checkpoint_path, dataset_path, spect_scaler_path
    """
    options_to_change = [
        {"section": "TRAIN", "option": "device", "value": device},
        path_option_to_change
    ]
    toml_path = specific_config(
        config_type="train",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.train.model)
    results_path = vak.common.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()

    with pytest.raises(FileNotFoundError):
        vak.train.frame_classification.train_frame_classification_model(
            model_name=cfg.train.model,
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


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"section": "TRAIN", "option": "dataset_path", "value": '/obviously/doesnt/exist/dataset-dir'},
        {"section": "TRAIN", "option": "root_results_dir", "value": '/obviously/doesnt/exist/results/'},
    ]
)
def test_train_raises_not_a_directory(
    path_option_to_change, specific_config, device, tmp_path
):
    """Test that core.train raises NotADirectory
    when directory does not exist
    """
    options_to_change = [
        path_option_to_change,
        {"section": "TRAIN", "option": "device", "value": device},
    ]

    toml_path = specific_config(
        config_type="train",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.train.model)

    # mock behavior of cli.train, building `results_path` from config option `root_results_dir`
    results_path = cfg.train.root_results_dir / 'results-dir-timestamp'

    with pytest.raises(NotADirectoryError):
        vak.train.frame_classification.train_frame_classification_model(
            model_name=cfg.train.model,
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
