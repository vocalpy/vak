"""tests for vak.core.predict module"""
from pathlib import Path

import pandas as pd
import pytest

import vak.config
import vak.constants
import vak.core.predict


# written as separate function so we can re-use in tests/unit/test_cli/test_predict.py
def predict_output_matches_expected(output_dir, annot_csv_filename):
    annot_csv = output_dir.joinpath(annot_csv_filename)
    assert annot_csv.exists()

    return True


@pytest.mark.parametrize(
    "audio_format, spect_format, annot_format, save_net_outputs",
    [
        ("cbin", None, "notmat", False),
        ("wav", None, "birdsong-recognition-dataset", False),
        ("cbin", None, "notmat", True),
        ("wav", None, "birdsong-recognition-dataset", True),
    ],
)
def test_predict(
    audio_format,
    spect_format,
    annot_format,
    save_net_outputs,
    specific_config,
    tmp_path,
    model,
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
        model=model,
        audio_format=audio_format,
        annot_format=annot_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    model_config_map = vak.config.models.map_from_path(toml_path, cfg.predict.models)

    vak.core.predict(
        csv_path=cfg.predict.csv_path,
        checkpoint_path=cfg.predict.checkpoint_path,
        labelmap_path=cfg.predict.labelmap_path,
        model_config_map=model_config_map,
        window_size=cfg.dataloader.window_size,
        num_workers=cfg.predict.num_workers,
        spect_key=cfg.spect_params.spect_key,
        timebins_key=cfg.spect_params.timebins_key,
        spect_scaler_path=cfg.predict.spect_scaler_path,
        device=cfg.predict.device,
        annot_csv_filename=cfg.predict.annot_csv_filename,
        output_dir=cfg.predict.output_dir,
        min_segment_dur=cfg.predict.min_segment_dur,
        majority_vote=cfg.predict.majority_vote,
        save_net_outputs=cfg.predict.save_net_outputs,
    )

    assert predict_output_matches_expected(output_dir, cfg.predict.annot_csv_filename)
    if save_net_outputs:
        net_outputs = sorted(
            Path(output_dir).glob(f"*{vak.constants.NET_OUTPUT_SUFFIX}")
        )

        vak_df = pd.read_csv(cfg.predict.csv_path)
        for spect_path in vak_df.spect_path.values:
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
        {"section": "PREDICT", "option": "csv_path", "value": '/obviously/doesnt/exist/dataset.csv'},
        {"section": "PREDICT", "option": "spect_scaler_path", "value": '/obviously/doesnt/exist/SpectScaler'},
    ]
)
def test_predict_raises_file_not_found(
    path_option_to_change,
    specific_config,
    tmp_path,
    device
):
    """Test that core.eval raises FileNotFoundError
    when `csv_path` does not exist."""
    output_dir = tmp_path.joinpath(
        f"test_predict_cbin_notmat_invalid_csv_path"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "PREDICT", "option": "output_dir", "value": str(output_dir)},
        {"section": "PREDICT", "option": "device", "value": device},
        path_option_to_change,
    ]
    toml_path = specific_config(
        config_type="predict",
        model="teenytweetynet",
        audio_format="cbin",
        annot_format="notmat",
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    model_config_map = vak.config.models.map_from_path(toml_path, cfg.predict.models)

    with pytest.raises(FileNotFoundError):
        vak.core.predict(
            csv_path=cfg.predict.csv_path,
            checkpoint_path=cfg.predict.checkpoint_path,
            labelmap_path=cfg.predict.labelmap_path,
            model_config_map=model_config_map,
            window_size=cfg.dataloader.window_size,
            num_workers=cfg.predict.num_workers,
            spect_key=cfg.spect_params.spect_key,
            timebins_key=cfg.spect_params.timebins_key,
            spect_scaler_path=cfg.predict.spect_scaler_path,
            device=cfg.predict.device,
            annot_csv_filename=cfg.predict.annot_csv_filename,
            output_dir=cfg.predict.output_dir,
            min_segment_dur=cfg.predict.min_segment_dur,
            majority_vote=cfg.predict.majority_vote,
            save_net_outputs=cfg.predict.save_net_outputs,
        )


def test_predict_raises_not_a_directory(
    specific_config,
    device
):
    """Test that core.eval raises NotADirectory
    when ``output_dir`` does not exist
    """
    options_to_change = [
        {"section": "PREDICT", "option": "output_dir", "value": '/obviously/does/not/exist/output'},
        {"section": "PREDICT", "option": "device", "value": device},
    ]
    toml_path = specific_config(
        config_type="predict",
        model="teenytweetynet",
        audio_format="cbin",
        annot_format="notmat",
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config_map = vak.config.models.map_from_path(toml_path, cfg.predict.models)

    with pytest.raises(NotADirectoryError):
        vak.core.predict(
            csv_path=cfg.predict.csv_path,
            checkpoint_path=cfg.predict.checkpoint_path,
            labelmap_path=cfg.predict.labelmap_path,
            model_config_map=model_config_map,
            window_size=cfg.dataloader.window_size,
            num_workers=cfg.predict.num_workers,
            spect_key=cfg.spect_params.spect_key,
            timebins_key=cfg.spect_params.timebins_key,
            spect_scaler_path=cfg.predict.spect_scaler_path,
            device=cfg.predict.device,
            annot_csv_filename=cfg.predict.annot_csv_filename,
            output_dir=cfg.predict.output_dir,
            min_segment_dur=cfg.predict.min_segment_dur,
            majority_vote=cfg.predict.majority_vote,
            save_net_outputs=cfg.predict.save_net_outputs,
        )
