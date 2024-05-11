"""tests for vak.predict module"""
from pathlib import Path

import pandas as pd
import pytest

import vak.config
import vak.common.constants
import vak.predict


# written as separate function so we can re-use in tests/unit/test_cli/test_predict.py
def assert_predict_output_matches_expected(output_dir, annot_csv_filename):
    annot_csv = output_dir.joinpath(annot_csv_filename)
    assert annot_csv.exists()


@pytest.mark.parametrize(
    "model_name, audio_format, spect_format, annot_format, save_net_outputs",
    [
        ("TweetyNet", "cbin", None, "notmat", False),
        ("TweetyNet", "cbin", None, "notmat", True),
    ],
)
def test_predict_with_frame_classification_model(
    model_name,
    audio_format,
    spect_format,
    annot_format,
    save_net_outputs,
        specific_config_toml_path,
    tmp_path,
    trainer_table,
):
    output_dir = tmp_path.joinpath(
        f"test_predict_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    keys_to_change = [
        {"table": "predict", "key": "output_dir", "value": str(output_dir)},
        {"table": "predict", "key": "trainer", "value": trainer_table},
        {"table": "predict", "key": "save_net_outputs", "value": save_net_outputs},
    ]
    toml_path = specific_config_toml_path(
        config_type="predict",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)

    vak.predict.frame_classification.predict_with_frame_classification_model(
        model_config=cfg.predict.model.asdict(),
        dataset_config=cfg.predict.dataset.asdict(),
        trainer_config=cfg.predict.trainer.asdict(),
        checkpoint_path=cfg.predict.checkpoint_path,
        labelmap_path=cfg.predict.labelmap_path,
        num_workers=cfg.predict.num_workers,
        timebins_key=cfg.prep.spect_params.timebins_key,
        frames_standardizer_path=cfg.predict.frames_standardizer_path,
        annot_csv_filename=cfg.predict.annot_csv_filename,
        output_dir=cfg.predict.output_dir,
        min_segment_dur=cfg.predict.min_segment_dur,
        majority_vote=cfg.predict.majority_vote,
        save_net_outputs=cfg.predict.save_net_outputs,
    )

    assert_predict_output_matches_expected(output_dir, cfg.predict.annot_csv_filename)
    if save_net_outputs:
        net_outputs = sorted(
            Path(output_dir).glob(f"*{vak.common.constants.NET_OUTPUT_SUFFIX}")
        )

        metadata = vak.datapipes.frame_classification.Metadata.from_dataset_path(cfg.predict.dataset.path)
        dataset_csv_path = cfg.predict.dataset.path / metadata.dataset_csv_filename
        dataset_df = pd.read_csv(dataset_csv_path)

        for spect_path in dataset_df.spect_path.values:
            net_output_spect_path = [
                net_output
                for net_output in net_outputs
                if net_output.name.startswith(Path(spect_path).stem)
            ]
            assert len(net_output_spect_path) == 1


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"table": "predict", "key": "checkpoint_path", "value": '/obviously/doesnt/exist/ckpt.pt'},
        {"table": "predict", "key": "labelmap_path", "value": '/obviously/doesnt/exist/labelmap.json'},
        {"table": "predict", "key": "frames_standardizer_path", "value": '/obviously/doesnt/exist/FramesStandardizer'},
    ]
)
def test_predict_with_frame_classification_model_raises_file_not_found(
    path_option_to_change,
        specific_config_toml_path,
    tmp_path,
    trainer_table
):
    """Test that core.eval raises FileNotFoundError
    when `dataset_path` does not exist."""
    output_dir = tmp_path.joinpath(
        f"test_predict_cbin_notmat_invalid_dataset_path"
    )
    output_dir.mkdir()

    keys_to_change = [
        {"table": "predict", "key": "output_dir", "value": str(output_dir)},
        {"table": "predict", "key": "trainer", "value": trainer_table},
        path_option_to_change,
    ]
    toml_path = specific_config_toml_path(
        config_type="predict",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)

    with pytest.raises(FileNotFoundError):
        vak.predict.frame_classification.predict_with_frame_classification_model(
            model_config=cfg.predict.model.asdict(),
            dataset_config=cfg.predict.dataset.asdict(),
            trainer_config=cfg.predict.trainer.asdict(),
            checkpoint_path=cfg.predict.checkpoint_path,
            labelmap_path=cfg.predict.labelmap_path,
            num_workers=cfg.predict.num_workers,
            timebins_key=cfg.prep.spect_params.timebins_key,
            frames_standardizer_path=cfg.predict.frames_standardizer_path,
            annot_csv_filename=cfg.predict.annot_csv_filename,
            output_dir=cfg.predict.output_dir,
            min_segment_dur=cfg.predict.min_segment_dur,
            majority_vote=cfg.predict.majority_vote,
            save_net_outputs=cfg.predict.save_net_outputs,
        )


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"table": "predict", "key": ["dataset", "path"], "value": '/obviously/doesnt/exist/dataset-dir'},
        {"table": "predict", "key": "output_dir", "value": '/obviously/does/not/exist/output'},
    ]
)
def test_predict_with_frame_classification_model_raises_not_a_directory(
    path_option_to_change,
        specific_config_toml_path,
    trainer_table,
    tmp_path,
):
    """Test that core.eval raises NotADirectory
    when ``output_dir`` does not exist
    """
    keys_to_change = [
        path_option_to_change,
        {"table": "predict", "key": "trainer", "value": trainer_table},
    ]

    if path_option_to_change["key"] != "output_dir":
        # need to make sure output_dir *does* exist
        # so we don't detect spurious NotADirectoryError and assume test passes
        output_dir = tmp_path.joinpath(
            f"test_predict_raises_not_a_directory"
        )
        output_dir.mkdir()
        keys_to_change.append(
            {"table": "predict", "key": "output_dir", "value": str(output_dir)}
        )

    toml_path = specific_config_toml_path(
        config_type="predict",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)

    with pytest.raises(NotADirectoryError):
        vak.predict.frame_classification.predict_with_frame_classification_model(
            model_config=cfg.predict.model.asdict(),
            dataset_config=cfg.predict.dataset.asdict(),
            trainer_config=cfg.predict.trainer.asdict(),
            checkpoint_path=cfg.predict.checkpoint_path,
            labelmap_path=cfg.predict.labelmap_path,
            num_workers=cfg.predict.num_workers,
            timebins_key=cfg.prep.spect_params.timebins_key,
            frames_standardizer_path=cfg.predict.frames_standardizer_path,
            annot_csv_filename=cfg.predict.annot_csv_filename,
            output_dir=cfg.predict.output_dir,
            min_segment_dur=cfg.predict.min_segment_dur,
            majority_vote=cfg.predict.majority_vote,
            save_net_outputs=cfg.predict.save_net_outputs,
        )
