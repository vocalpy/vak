"""tests for vak.eval.frame_classification module"""
import pytest

import vak.config
import vak.common.constants
import vak.common.paths
import vak.eval.frame_classification


# written as separate function so we can re-use in tests/unit/test_cli/test_eval.py
def assert_eval_output_matches_expected(model_name, output_dir):
    eval_csv = sorted(output_dir.glob(f"eval_{model_name}*csv"))
    assert len(eval_csv) == 1


# -- we do eval with all possible configurations of post_tfm_kwargs
POST_TFM_KWARGS = [
    # default, will use ToLabels
    None,
    # no cleanup but uses ToLabelsWithPostprocessing
    {'majority_vote': False, 'min_segment_dur': None},
    # use ToLabelsWithPostprocessing with *just* majority_vote
    {'majority_vote': True, 'min_segment_dur': None},
    # use ToLabelsWithPostprocessing with *just* min_segment_dur
    {'majority_vote': False, 'min_segment_dur': 0.002},
    # use ToLabelsWithPostprocessing with majority_vote *and* min_segment_dur
    {'majority_vote': True, 'min_segment_dur': 0.002},
]


@pytest.fixture(params=POST_TFM_KWARGS)
def post_tfm_kwargs(request):
    return request.param


@pytest.mark.parametrize(
    "model_name, audio_format, spect_format, annot_format",
    [
        ("TweetyNet", "cbin", None, "notmat"),
    ],
)
def test_eval_frame_classification_model(
        model_name,
        audio_format,
        spect_format,
        annot_format,
        specific_config_toml_path,
        tmp_path,
        trainer_table,
        post_tfm_kwargs
):
    output_dir = tmp_path.joinpath(
        f"test_eval_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    keys_to_change = [
        {"table": "eval", "key": "output_dir", "value": str(output_dir)},
        {"table": "eval", "key": "trainer", "value": trainer_table},
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

    vak.eval.frame_classification.eval_frame_classification_model(
        model_config=cfg.eval.model.asdict(),
        dataset_config=cfg.eval.dataset.asdict(),
        trainer_config=cfg.eval.trainer.asdict(),
        checkpoint_path=cfg.eval.checkpoint_path,
        labelmap_path=cfg.eval.labelmap_path,
        output_dir=cfg.eval.output_dir,
        num_workers=cfg.eval.num_workers,
        frames_standardizer_path=cfg.eval.frames_standardizer_path,
        post_tfm_kwargs=post_tfm_kwargs,
    )

    assert_eval_output_matches_expected(cfg.eval.model.name, output_dir)


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"table": "eval", "key": "checkpoint_path", "value": '/obviously/doesnt/exist/ckpt.pt'},
        {"table": "eval", "key": "labelmap_path", "value": '/obviously/doesnt/exist/labelmap.json'},
        {"table": "eval", "key": "frames_standardizer_path", "value": '/obviously/doesnt/exist/FramesStandardizer'},
    ]
)
def test_eval_frame_classification_model_raises_file_not_found(
    path_option_to_change,
        specific_config_toml_path,
    tmp_path,
    trainer_table
):
    """Test that core.eval raises FileNotFoundError
    when one of the following does not exist:
    checkpoint_path, labelmap_path, dataset_path, frames_standardizer_path
    """
    output_dir = tmp_path.joinpath(
        f"test_eval_cbin_notmat_invalid_dataset_path"
    )
    output_dir.mkdir()

    keys_to_change = [
        {"table": "eval", "key": "output_dir", "value": str(output_dir)},
        {"table": "eval", "key": "trainer", "value": trainer_table},
        path_option_to_change,
    ]

    toml_path = specific_config_toml_path(
        config_type="eval",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)
    with pytest.raises(FileNotFoundError):
        vak.eval.frame_classification.eval_frame_classification_model(
            model_config=cfg.eval.model.asdict(),
            dataset_config=cfg.eval.dataset.asdict(),
            trainer_config=cfg.eval.trainer.asdict(),
            checkpoint_path=cfg.eval.checkpoint_path,
            labelmap_path=cfg.eval.labelmap_path,
            output_dir=cfg.eval.output_dir,
            num_workers=cfg.eval.num_workers,
            frames_standardizer_path=cfg.eval.frames_standardizer_path,
        )


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"table": "eval", "key": ["dataset","path"], "value": '/obviously/doesnt/exist/dataset-dir'},
        {"table": "eval", "key": "output_dir", "value": '/obviously/does/not/exist/output'},
    ]
)
def test_eval_frame_classification_model_raises_not_a_directory(
    path_option_to_change,
        specific_config_toml_path,
    trainer_table,
    tmp_path,
):
    """Test that core.eval raises NotADirectory
    when directories don't exist
    """
    keys_to_change = [
        path_option_to_change,
        {"table": "eval", "key": "trainer", "value": trainer_table},
    ]

    if path_option_to_change["key"] != "output_dir":
        # need to make sure output_dir *does* exist
        # so we don't detect spurious NotADirectoryError and assume test passes
        output_dir = tmp_path.joinpath(
            f"test_eval_raises_not_a_directory"
        )
        output_dir.mkdir()
        keys_to_change.append(
            {"table": "eval", "key": "output_dir", "value": str(output_dir)}
        )

    toml_path = specific_config_toml_path(
        config_type="eval",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)
    with pytest.raises(NotADirectoryError):
        vak.eval.frame_classification.eval_frame_classification_model(
            model_config=cfg.eval.model.asdict(),
            dataset_config=cfg.eval.dataset.asdict(),
            trainer_config=cfg.eval.trainer.asdict(),
            checkpoint_path=cfg.eval.checkpoint_path,
            labelmap_path=cfg.eval.labelmap_path,
            output_dir=cfg.eval.output_dir,
            num_workers=cfg.eval.num_workers,
            frames_standardizer_path=cfg.eval.frames_standardizer_path,
        )
