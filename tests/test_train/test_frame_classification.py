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

    if cfg.train.standardize_frames or cfg.train.frames_standardizer_path:
        assert results_path.joinpath("FramesStandardizer").exists()
    else:
        assert not results_path.joinpath("FramesStandardizer").exists()

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
        ("TweetyNet", None, "mat", "yarden"),
    ],
)
def test_train_frame_classification_model(
    model_name, audio_format, spect_format, annot_format, specific_config_toml_path, tmp_path, trainer_table
):
    results_path = vak.common.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()
    keys_to_change = [
        {"table": "train", "key": "trainer", "value": trainer_table},
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

    vak.train.frame_classification.train_frame_classification_model(
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

    assert_train_output_matches_expected(cfg, cfg.train.model.name, results_path)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name, audio_format, spect_format, annot_format",
    [
        ("TweetyNet", "cbin", None, "notmat"),
        ("TweetyNet", None, "mat", "yarden"),
    ],
)
def test_continue_training(
    model_name, audio_format, spect_format, annot_format, specific_config_toml_path, tmp_path, trainer_table
):
    results_path = vak.common.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()
    keys_to_change = [
        {"table": "train", "key": "trainer", "value": trainer_table},
        {"table": "train", "key": "root_results_dir", "value": str(results_path)}
    ]
    toml_path = specific_config_toml_path(
        config_type="train_continue",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)

    vak.train.frame_classification.train_frame_classification_model(
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

    assert_train_output_matches_expected(cfg, cfg.train.model.name, results_path)


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"table": "train", "key": "checkpoint_path", "value": '/obviously/doesnt/exist/ckpt.pt'},
        {"table": "train", "key": "frames_standardizer_path", "value": '/obviously/doesnt/exist/FramesStandardizer'},
    ]
)
def test_train_raises_file_not_found(
    path_option_to_change, specific_config_toml_path, tmp_path, trainer_table
):
    """Test that pre-conditions in `vak.train` raise FileNotFoundError
    when one of the following does not exist:
    checkpoint_path, dataset_path, frames_standardizer_path
    """
    keys_to_change = [
        {"table": "train", "key": "trainer", "value": trainer_table},
        path_option_to_change
    ]
    toml_path = specific_config_toml_path(
        config_type="train",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)
    results_path = vak.common.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()

    with pytest.raises(FileNotFoundError):
        vak.train.frame_classification.train_frame_classification_model(
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


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"table": "train", "key": ["dataset", "path"], "value": '/obviously/doesnt/exist/dataset-dir'},
        {"table": "train", "key": "root_results_dir", "value": '/obviously/doesnt/exist/results/'},
    ]
)
def test_train_raises_not_a_directory(
    path_option_to_change, specific_config_toml_path, trainer_table, tmp_path
):
    """Test that core.train raises NotADirectory
    when directory does not exist
    """
    keys_to_change = [
        path_option_to_change,
        {"table": "train", "key": "trainer", "value": trainer_table},
    ]

    toml_path = specific_config_toml_path(
        config_type="train",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)

    # mock behavior of cli.train, building `results_path` from config option `root_results_dir`
    results_path = cfg.train.root_results_dir / 'results-dir-timestamp'

    with pytest.raises(NotADirectoryError):
        vak.train.frame_classification.train_frame_classification_model(
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
