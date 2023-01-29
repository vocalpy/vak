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
    "audio_format, spect_format, annot_format",
    [
        ("cbin", None, "notmat"),
    ],
)
def test_eval(
        audio_format,
        spect_format,
        annot_format,
        specific_config,
        tmp_path,
        model,
        device,
        post_tfm_kwargs
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
        post_tfm_kwargs=post_tfm_kwargs,
    )

    assert eval_output_matches_expected(model_config_map, output_dir)


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"section": "EVAL", "option": "checkpoint_path", "value": '/obviously/doesnt/exist/ckpt.pt'},
        {"section": "EVAL", "option": "labelmap_path", "value": '/obviously/doesnt/exist/labelmap.json'},
        {"section": "EVAL", "option": "csv_path", "value": '/obviously/doesnt/exist/dataset.csv'},
        {"section": "EVAL", "option": "spect_scaler_path", "value": '/obviously/doesnt/exist/SpectScaler'},
    ]
)
def test_eval_raises_file_not_found(
    path_option_to_change,
    specific_config,
    tmp_path,
    device
):
    """Test that core.eval raises FileNotFoundError
    when one of the following does not exist:
    checkpoint_path, labelmap_path, csv_path, spect_scaler_path
    """
    output_dir = tmp_path.joinpath(
        f"test_eval_cbin_notmat_invalid_csv_path"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "EVAL", "option": "output_dir", "value": str(output_dir)},
        {"section": "EVAL", "option": "device", "value": device},
        path_option_to_change,
    ]

    toml_path = specific_config(
        config_type="eval",
        model="teenytweetynet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config_map = vak.config.models.map_from_path(toml_path, cfg.eval.models)
    with pytest.raises(FileNotFoundError):
        vak.core.eval(
            csv_path=cfg.eval.csv_path,
            model_config_map=model_config_map,
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


def test_eval_raises_not_a_directory(
    specific_config,
    device
):
    """Test that core.eval raises NotADirectory
    when ``output_dir`` does not exist
    """
    options_to_change = [
        {"section": "EVAL", "option": "output_dir", "value": '/obviously/does/not/exist/output'},
        {"section": "EVAL", "option": "device", "value": device},
    ]

    toml_path = specific_config(
        config_type="eval",
        model="teenytweetynet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config_map = vak.config.models.map_from_path(toml_path, cfg.eval.models)
    with pytest.raises(NotADirectoryError):
        vak.core.eval(
            csv_path=cfg.eval.csv_path,
            model_config_map=model_config_map,
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
