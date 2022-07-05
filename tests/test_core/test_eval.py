"""tests for vak.core.eval module"""
import pytest

import vak.config
import vak.constants
import vak.paths
import vak.core.eval


# written as separate function so we can re-use in tests/unit/test_cli/test_eval.py
def eval_output_matches_expected(model_config_map, output_dir):
    for model_name in model_config_map.keys():
        eval_csv = sorted(output_dir.glob(f"eval_{model_name}*csv"))
        assert len(eval_csv) == 1

    return True


@pytest.mark.parametrize(
    "audio_format, spect_format, annot_format",
    [
        ("cbin", None, "notmat"),
    ],
)
def test_eval(
    audio_format, spect_format, annot_format, specific_config, tmp_path, model, device
):
    output_dir = tmp_path.joinpath(
        f"test_eval_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "EVAL", "option": "output_dir", "value": str(output_dir)},
        {"section": "EVAL", "option": "device", "value": device},
    ]

    toml_path = specific_config(
        config_type="eval",
        model=model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config_map = vak.config.models.map_from_path(toml_path, cfg.eval.models)

    vak.core.eval(
        cfg.eval.csv_path,
        model_config_map,
        checkpoint_path=cfg.eval.checkpoint_path,
        labelmap_path=cfg.eval.labelmap_path,
        output_dir=cfg.eval.output_dir,
        window_size=cfg.dataloader.window_size,
        num_workers=cfg.eval.num_workers,
        spect_scaler_path=cfg.eval.spect_scaler_path,
        spect_key=cfg.spect_params.spect_key,
        timebins_key=cfg.spect_params.timebins_key,
        device=cfg.eval.device,
    )

    assert eval_output_matches_expected(model_config_map, output_dir)
