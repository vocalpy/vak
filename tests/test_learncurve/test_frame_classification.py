"""tests for vak.learncurve.frame_classification module"""
import pytest

import vak.config
import vak.common.constants
import vak.learncurve
import vak.common.paths


def assert_learncurve_output_matches_expected(cfg, model_name, results_path):
    assert results_path.joinpath("learning_curve.csv").exists()

    for train_set_dur in cfg.prep.train_set_durs:
        train_set_dur_root = results_path / vak.learncurve.dirname.train_dur_dirname(train_set_dur)
        assert train_set_dur_root.exists()

        for replicate_num in range(1, cfg.prep.num_replicates + 1):
            replicate_path = train_set_dur_root / vak.learncurve.dirname.replicate_dirname(replicate_num)
            assert replicate_path.exists()

            assert replicate_path.joinpath("labelmap.json").exists()

            if cfg.learncurve.normalize_spectrograms:
                assert replicate_path.joinpath("StandardizeSpect").exists()

            eval_csv = sorted(replicate_path.glob(f"eval_{model_name}*csv"))
            assert len(eval_csv) == 1

            model_path = replicate_path.joinpath(model_name)
            assert model_path.exists()

            tensorboard_log = sorted(
                replicate_path.glob(f"lightning_logs/**/*events*")
            )
            assert len(tensorboard_log) == 1

            checkpoints_path = model_path.joinpath("checkpoints")
            assert checkpoints_path.exists()
            assert checkpoints_path.joinpath("checkpoint.pt").exists()
            if cfg.learncurve.val_step is not None:
                assert checkpoints_path.joinpath(
                    "max-val-acc-checkpoint.pt"
                ).exists()


@pytest.mark.slow
@pytest.mark.parametrize(
    'model_name, audio_format, annot_format',
    [
        ("TweetyNet", "cbin", "notmat"),
    ]
)
def test_learning_curve_for_frame_classification_model(
        model_name, audio_format, annot_format, specific_config, tmp_path, device):
    options_to_change = {"section": "LEARNCURVE", "option": "device", "value": device}

    toml_path = specific_config(
        config_type="learncurve",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        options_to_change=options_to_change,
    )

    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.learncurve.model)
    results_path = vak.common.paths.generate_results_dir_name_as_path(tmp_path)
    results_path.mkdir()

    vak.learncurve.frame_classification.learning_curve_for_frame_classification_model(
        model_name=cfg.learncurve.model,
        model_config=model_config,
        dataset_path=cfg.learncurve.dataset_path,
        batch_size=cfg.learncurve.batch_size,
        num_epochs=cfg.learncurve.num_epochs,
        num_workers=cfg.learncurve.num_workers,
        train_transform_params=cfg.learncurve.train_transform_params,
        train_dataset_params=cfg.learncurve.train_dataset_params,
        val_transform_params=cfg.learncurve.val_transform_params,
        val_dataset_params=cfg.learncurve.val_dataset_params,
        results_path=results_path,
        post_tfm_kwargs=cfg.learncurve.post_tfm_kwargs,
        normalize_spectrograms=cfg.learncurve.normalize_spectrograms,
        shuffle=cfg.learncurve.shuffle,
        val_step=cfg.learncurve.val_step,
        ckpt_step=cfg.learncurve.ckpt_step,
        patience=cfg.learncurve.patience,
        device=cfg.learncurve.device,
    )

    assert_learncurve_output_matches_expected(cfg, cfg.learncurve.model, results_path)


@pytest.mark.parametrize(
    'dir_option_to_change',
    [
        {"section": "LEARNCURVE", "option": "root_results_dir", "value": '/obviously/does/not/exist/results/'},
        {"section": "LEARNCURVE", "option": "dataset_path", "value": '/obviously/doesnt/exist/dataset-dir'},
    ]
)
def test_learncurve_raises_not_a_directory(dir_option_to_change,
                                           specific_config,
                                           tmp_path, device):
    """Test that core.learncurve.learning_curve raises NotADirectoryError
    when the following directories do not exist:
    results_path, previous_run_path, dataset_path
    """
    options_to_change = [
        {"section": "LEARNCURVE", "option": "device", "value": device},
        dir_option_to_change
    ]
    toml_path = specific_config(
        config_type="learncurve",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.learncurve.model)
    # mock behavior of cli.learncurve, building `results_path` from config option `root_results_dir`
    results_path = cfg.learncurve.root_results_dir / 'results-dir-timestamp'

    with pytest.raises(NotADirectoryError):
        vak.learncurve.frame_classification.learning_curve_for_frame_classification_model(
            model_name=cfg.learncurve.model,
            model_config=model_config,
            dataset_path=cfg.learncurve.dataset_path,
            batch_size=cfg.learncurve.batch_size,
            num_epochs=cfg.learncurve.num_epochs,
            num_workers=cfg.learncurve.num_workers,
            train_transform_params=cfg.learncurve.train_transform_params,
            train_dataset_params=cfg.learncurve.train_dataset_params,
            val_transform_params=cfg.learncurve.val_transform_params,
            val_dataset_params=cfg.learncurve.val_dataset_params,
            results_path=results_path,
            post_tfm_kwargs=cfg.learncurve.post_tfm_kwargs,
            normalize_spectrograms=cfg.learncurve.normalize_spectrograms,
            shuffle=cfg.learncurve.shuffle,
            val_step=cfg.learncurve.val_step,
            ckpt_step=cfg.learncurve.ckpt_step,
            patience=cfg.learncurve.patience,
            device=cfg.learncurve.device,
        )
