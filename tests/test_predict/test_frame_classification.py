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
        ("TweetyNet", "wav", None, "birdsong-recognition-dataset", False),
        ("TweetyNet", "cbin", None, "notmat", True),
        ("TweetyNet", "wav", None, "birdsong-recognition-dataset", True),
    ],
)
def test_predict_with_frame_classification_model(
    model_name,
    audio_format,
    spect_format,
    annot_format,
    save_net_outputs,
    specific_config,
    tmp_path,
    device,
):
    output_dir = tmp_path.joinpath(
        f"test_predict_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "PREDICT", "option": "output_dir", "value": str(output_dir)},
        {"section": "PREDICT", "option": "device", "value": device},
        {"section": "PREDICT", "option": "save_net_outputs", "value": save_net_outputs},
    ]
    toml_path = specific_config(
        config_type="predict",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.predict.model)

    vak.predict.frame_classification.predict_with_frame_classification_model(
        model_name=cfg.predict.model,
        model_config=model_config,
        dataset_path=cfg.predict.dataset_path,
        checkpoint_path=cfg.predict.checkpoint_path,
        labelmap_path=cfg.predict.labelmap_path,
        num_workers=cfg.predict.num_workers,
        transform_params=cfg.predict.transform_params,
        dataset_params=cfg.predict.dataset_params,
        timebins_key=cfg.spect_params.timebins_key,
        spect_scaler_path=cfg.predict.spect_scaler_path,
        device=cfg.predict.device,
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

        metadata = vak.datasets.frame_classification.Metadata.from_dataset_path(cfg.predict.dataset_path)
        dataset_csv_path = cfg.predict.dataset_path / metadata.dataset_csv_filename
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
        {"section": "PREDICT", "option": "checkpoint_path", "value": '/obviously/doesnt/exist/ckpt.pt'},
        {"section": "PREDICT", "option": "labelmap_path", "value": '/obviously/doesnt/exist/labelmap.json'},
        {"section": "PREDICT", "option": "spect_scaler_path", "value": '/obviously/doesnt/exist/SpectScaler'},
    ]
)
def test_predict_with_frame_classification_model_raises_file_not_found(
    path_option_to_change,
    specific_config,
    tmp_path,
    device
):
    """Test that core.eval raises FileNotFoundError
    when `dataset_path` does not exist."""
    output_dir = tmp_path.joinpath(
        f"test_predict_cbin_notmat_invalid_dataset_path"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "PREDICT", "option": "output_dir", "value": str(output_dir)},
        {"section": "PREDICT", "option": "device", "value": device},
        path_option_to_change,
    ]
    toml_path = specific_config(
        config_type="predict",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.predict.model)

    with pytest.raises(FileNotFoundError):
        vak.predict.frame_classification.predict_with_frame_classification_model(
            model_name=cfg.predict.model,
            model_config=model_config,
            dataset_path=cfg.predict.dataset_path,
            checkpoint_path=cfg.predict.checkpoint_path,
            labelmap_path=cfg.predict.labelmap_path,
            num_workers=cfg.predict.num_workers,
            transform_params=cfg.predict.transform_params,
            dataset_params=cfg.predict.dataset_params,
            timebins_key=cfg.spect_params.timebins_key,
            spect_scaler_path=cfg.predict.spect_scaler_path,
            device=cfg.predict.device,
            annot_csv_filename=cfg.predict.annot_csv_filename,
            output_dir=cfg.predict.output_dir,
            min_segment_dur=cfg.predict.min_segment_dur,
            majority_vote=cfg.predict.majority_vote,
            save_net_outputs=cfg.predict.save_net_outputs,
        )


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"section": "PREDICT", "option": "dataset_path", "value": '/obviously/doesnt/exist/dataset-dir'},
        {"section": "PREDICT", "option": "output_dir", "value": '/obviously/does/not/exist/output'},
    ]
)
def test_predict_with_frame_classification_model_raises_not_a_directory(
    path_option_to_change,
    specific_config,
    device,
    tmp_path,
):
    """Test that core.eval raises NotADirectory
    when ``output_dir`` does not exist
    """
    options_to_change = [
        path_option_to_change,
        {"section": "PREDICT", "option": "device", "value": device},
    ]

    if path_option_to_change["option"] != "output_dir":
        # need to make sure output_dir *does* exist
        # so we don't detect spurious NotADirectoryError and assume test passes
        output_dir = tmp_path.joinpath(
            f"test_predict_raises_not_a_directory"
        )
        output_dir.mkdir()
        options_to_change.append(
            {"section": "PREDICT", "option": "output_dir", "value": str(output_dir)}
        )

    toml_path = specific_config(
        config_type="predict",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.predict.model)

    with pytest.raises(NotADirectoryError):
        vak.predict.frame_classification.predict_with_frame_classification_model(
            model_name=cfg.predict.model,
            model_config=model_config,
            dataset_path=cfg.predict.dataset_path,
            checkpoint_path=cfg.predict.checkpoint_path,
            labelmap_path=cfg.predict.labelmap_path,
            num_workers=cfg.predict.num_workers,
            transform_params=cfg.predict.transform_params,
            dataset_params=cfg.predict.dataset_params,
            timebins_key=cfg.spect_params.timebins_key,
            spect_scaler_path=cfg.predict.spect_scaler_path,
            device=cfg.predict.device,
            annot_csv_filename=cfg.predict.annot_csv_filename,
            output_dir=cfg.predict.output_dir,
            min_segment_dur=cfg.predict.min_segment_dur,
            majority_vote=cfg.predict.majority_vote,
            save_net_outputs=cfg.predict.save_net_outputs,
        )
