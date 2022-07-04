"""tests for vak.core.train module"""
import pytest

import vak.config
import vak.constants
import vak.paths
import vak.core.train


def train_output_matches_expected(cfg, model_config_map, results_path):
    assert results_path.joinpath("labelmap.json").exists()

    if cfg.train.normalize_spectrograms or cfg.train.spect_scaler_path:
        assert results_path.joinpath("StandardizeSpect").exists()
    else:
        assert not results_path.joinpath("StandardizeSpect").exists()

    for model_name in model_config_map.keys():
        model_path = results_path.joinpath(model_name)
        assert model_path.exists()

        tensorboard_log = sorted(model_path.glob(f"events.out.tfevents.*{model_name}"))
        assert len(tensorboard_log) == 1

        checkpoints_path = model_path.joinpath("checkpoints")
        assert checkpoints_path.exists()
        assert checkpoints_path.joinpath("checkpoint.pt").exists()
        if cfg.train.val_step is not None:
            assert checkpoints_path.joinpath("max-val-acc-checkpoint.pt").exists()

    return True


@pytest.mark.parametrize(
    "audio_format, spect_format, annot_format",
    [
        ("cbin", None, "notmat"),
        ("wav", None, "birdsong-recognition-dataset"),
        (None, "mat", "yarden"),
    ],
)
def test_train(
    audio_format, spect_format, annot_format, specific_config, tmp_path, model, device
):
    options_to_change = {"section": "TRAIN", "option": "device", "value": device}
    toml_path = specific_config(
        config_type="train",
        model=model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config_map = vak.config.models.map_from_path(toml_path, cfg.train.models)
    results_path = vak.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()

    vak.core.train(
        model_config_map,
        cfg.train.csv_path,
        cfg.dataloader.window_size,
        cfg.train.batch_size,
        cfg.train.num_epochs,
        cfg.train.num_workers,
        labelset=cfg.prep.labelset,
        results_path=results_path,
        spect_key=cfg.spect_params.spect_key,
        timebins_key=cfg.spect_params.timebins_key,
        normalize_spectrograms=cfg.train.normalize_spectrograms,
        shuffle=cfg.train.shuffle,
        val_step=cfg.train.val_step,
        ckpt_step=cfg.train.ckpt_step,
        patience=cfg.train.patience,
        device=cfg.train.device,
    )

    assert train_output_matches_expected(cfg, model_config_map, results_path)


@pytest.mark.parametrize(
    "audio_format, spect_format, annot_format",
    [
        ("cbin", None, "notmat"),
        ("wav", None, "birdsong-recognition-dataset"),
        (None, "mat", "yarden"),
    ],
)
def test_continue_training(
    audio_format, spect_format, annot_format, specific_config, tmp_path, model, device
):
    options_to_change = {"section": "TRAIN", "option": "device", "value": device}
    toml_path = specific_config(
        config_type="train_continue",
        model=model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config_map = vak.config.models.map_from_path(toml_path, cfg.train.models)
    results_path = vak.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()

    vak.core.train(
        model_config_map=model_config_map,
        csv_path=cfg.train.csv_path,
        window_size=cfg.dataloader.window_size,
        batch_size=cfg.train.batch_size,
        num_epochs=cfg.train.num_epochs,
        num_workers=cfg.train.num_workers,
        labelmap_path=cfg.train.labelmap_path,
        spect_scaler_path=cfg.train.spect_scaler_path,
        results_path=results_path,
        spect_key=cfg.spect_params.spect_key,
        timebins_key=cfg.spect_params.timebins_key,
        normalize_spectrograms=cfg.train.normalize_spectrograms,
        shuffle=cfg.train.shuffle,
        val_step=cfg.train.val_step,
        ckpt_step=cfg.train.ckpt_step,
        patience=cfg.train.patience,
        device=cfg.train.device,
    )

    assert train_output_matches_expected(cfg, model_config_map, results_path)
