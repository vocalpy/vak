"""Tests for vak.train.parametric_umap module"""
import pathlib

import pytest

import vak.config
import vak.common.constants
import vak.common.paths
import vak.train


def assert_train_output_matches_expected(cfg: vak.config.config.Config, model_name: str,
                                         results_path: pathlib.Path):
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
        assert checkpoints_path.joinpath("min-val-loss-checkpoint.pt").exists()


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name, audio_format, spect_format, annot_format",
    [
        ("ConvEncoderUMAP", "cbin", None, "notmat"),
    ],
)
def test_train_parametric_umap_model(
    model_name, audio_format, spect_format, annot_format,
        specific_config_toml_path, tmp_path, device
):
    results_path = vak.common.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()
    keys_to_change = [
        {"table": "train", "key": "device", "value": device},
        {"table": "train", "key": "root_results_dir", "value": str(results_path)}
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

    vak.train.parametric_umap.train_parametric_umap_model(
        model_name=cfg.train.model.name,
        model_config=cfg.train.model.to_dict(),
        dataset_path=cfg.train.dataset.path,
        batch_size=cfg.train.batch_size,
        num_epochs=cfg.train.num_epochs,
        num_workers=cfg.train.num_workers,
        train_transform_params=cfg.train.train_transform_params,
        train_dataset_params=cfg.train.train_dataset_params,
        val_transform_params=cfg.train.val_transform_params,
        val_dataset_params=cfg.train.val_dataset_params,
        checkpoint_path=cfg.train.checkpoint_path,
        results_path=results_path,
        shuffle=cfg.train.shuffle,
        val_step=cfg.train.val_step,
        ckpt_step=cfg.train.ckpt_step,
        device=cfg.train.device,
    )

    assert_train_output_matches_expected(cfg, cfg.train.model.name, results_path)


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"table": "train", "key": "checkpoint_path", "value": '/obviously/doesnt/exist/ckpt.pt'},
    ]
)
def test_train_parametric_umap_model_raises_file_not_found(
    path_option_to_change, specific_config_toml_path, tmp_path, device
):
    """Test that pre-conditions in :func:`vak.train.parametric_umap.train_parametric_umap_model`
    raise FileNotFoundError when one of the following does not exist:
    checkpoint_path, dataset_path
    """
    keys_to_change = [
        {"table": "train", "key": "device", "value": device},
        path_option_to_change
    ]
    toml_path = specific_config_toml_path(
        config_type="train",
        model="ConvEncoderUMAP",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)
    results_path = vak.common.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()

    with pytest.raises(FileNotFoundError):
        vak.train.parametric_umap.train_parametric_umap_model(
            model_name=cfg.train.model.name,
            model_config=cfg.train.model.to_dict(),
            dataset_path=cfg.train.dataset.path,
            batch_size=cfg.train.batch_size,
            num_epochs=cfg.train.num_epochs,
            num_workers=cfg.train.num_workers,
            train_transform_params=cfg.train.train_transform_params,
            train_dataset_params=cfg.train.train_dataset_params,
            val_transform_params=cfg.train.val_transform_params,
            val_dataset_params=cfg.train.val_dataset_params,
            checkpoint_path=cfg.train.checkpoint_path,
            results_path=results_path,
            shuffle=cfg.train.shuffle,
            val_step=cfg.train.val_step,
            ckpt_step=cfg.train.ckpt_step,
            device=cfg.train.device,
        )


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"table": "train", "key": ["dataset", "path"], "value": '/obviously/doesnt/exist/dataset-dir'},
        {"table": "train", "key": "root_results_dir", "value": '/obviously/doesnt/exist/results/'},
    ]
)
def test_train_parametric_umap_model_raises_not_a_directory(
    path_option_to_change, specific_config_toml_path, device, tmp_path
):
    """Test that core.train raises NotADirectory
    when directory does not exist
    """
    keys_to_change = [
        path_option_to_change,
        {"table": "train", "key": "device", "value": device},
    ]

    toml_path = specific_config_toml_path(
        config_type="train",
        model="ConvEncoderUMAP",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)
    model_config = cfg.train.model.to_dict()

    # mock behavior of cli.train, building `results_path` from config option `root_results_dir`
    results_path = cfg.train.root_results_dir / 'results-dir-timestamp'

    with pytest.raises(NotADirectoryError):
        vak.train.parametric_umap.train_parametric_umap_model(
            model_name=cfg.train.model,
            model_config=model_config,
            dataset_path=cfg.train.dataset.path,
            batch_size=cfg.train.batch_size,
            num_epochs=cfg.train.num_epochs,
            num_workers=cfg.train.num_workers,
            train_transform_params=cfg.train.train_transform_params,
            train_dataset_params=cfg.train.train_dataset_params,
            val_transform_params=cfg.train.val_transform_params,
            val_dataset_params=cfg.train.val_dataset_params,
            checkpoint_path=cfg.train.checkpoint_path,
            results_path=results_path,
            shuffle=cfg.train.shuffle,
            val_step=cfg.train.val_step,
            ckpt_step=cfg.train.ckpt_step,
            device=cfg.train.device,
        )
