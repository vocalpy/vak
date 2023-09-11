"""tests for vak.eval.parametric_umap module"""
import pytest

import vak.config
import vak.common.constants
import vak.common.paths
import vak.eval.frame_classification


def assert_eval_output_matches_expected(model_name, output_dir):
    eval_csv = sorted(output_dir.glob(f"eval_{model_name}*csv"))
    assert len(eval_csv) == 1


@pytest.mark.parametrize(
    "model_name, audio_format, spect_format, annot_format",
    [
        ("ConvEncoderUMAP", "cbin", None, "notmat"),
    ],
)
def test_eval_parametric_umap_model(
        model_name,
        audio_format,
        spect_format,
        annot_format,
        specific_config,
        tmp_path,
        device,
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
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.eval.model)

    vak.eval.parametric_umap.eval_parametric_umap_model(
        model_name=cfg.eval.model,
        model_config=model_config,
        dataset_path=cfg.eval.dataset_path,
        checkpoint_path=cfg.eval.checkpoint_path,
        output_dir=cfg.eval.output_dir,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.eval.num_workers,
        transform_params=cfg.eval.transform_params,
        dataset_params=cfg.eval.dataset_params,
        device=cfg.eval.device,
    )

    assert_eval_output_matches_expected(cfg.eval.model, output_dir)


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"section": "EVAL", "option": "checkpoint_path", "value": '/obviously/doesnt/exist/ckpt.pt'},
    ]
)
def test_eval_frame_classification_model_raises_file_not_found(
    path_option_to_change,
    specific_config,
    tmp_path,
    device
):
    """Test that :func:`vak.eval.parametric_umap.eval_parametric_umap_model`
    raises FileNotFoundError when expected"""
    output_dir = tmp_path.joinpath(
        f"test_eval_cbin_notmat_invalid_dataset_path"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "EVAL", "option": "output_dir", "value": str(output_dir)},
        {"section": "EVAL", "option": "device", "value": device},
        path_option_to_change,
    ]

    toml_path = specific_config(
        config_type="eval",
        model="ConvEncoderUMAP",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.eval.model)
    with pytest.raises(FileNotFoundError):
        vak.eval.parametric_umap.eval_parametric_umap_model(
            model_name=cfg.eval.model,
            model_config=model_config,
            dataset_path=cfg.eval.dataset_path,
            checkpoint_path=cfg.eval.checkpoint_path,
            output_dir=cfg.eval.output_dir,
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.num_workers,
            transform_params=cfg.eval.transform_params,
            dataset_params=cfg.eval.dataset_params,
            device=cfg.eval.device,
        )


@pytest.mark.parametrize(
    'path_option_to_change',
    [
        {"section": "EVAL", "option": "dataset_path", "value": '/obviously/doesnt/exist/dataset-dir'},
        {"section": "EVAL", "option": "output_dir", "value": '/obviously/does/not/exist/output'},
    ]
)
def test_eval_frame_classification_model_raises_not_a_directory(
    path_option_to_change,
    specific_config,
    device,
    tmp_path,
):
    """Test that :func:`vak.eval.parametric_umap.eval_parametric_umap_model`
    raises NotADirectoryError when expected"""
    options_to_change = [
        path_option_to_change,
        {"section": "EVAL", "option": "device", "value": device},
    ]

    if path_option_to_change["option"] != "output_dir":
        # need to make sure output_dir *does* exist
        # so we don't detect spurious NotADirectoryError and assume test passes
        output_dir = tmp_path.joinpath(
            f"test_eval_raises_not_a_directory"
        )
        output_dir.mkdir()
        options_to_change.append(
            {"section": "EVAL", "option": "output_dir", "value": str(output_dir)}
        )

    toml_path = specific_config(
        config_type="eval",
        model="ConvEncoderUMAP",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)
    model_config = vak.config.model.config_from_toml_path(toml_path, cfg.eval.model)
    with pytest.raises(NotADirectoryError):
        vak.eval.parametric_umap.eval_parametric_umap_model(
            model_name=cfg.eval.model,
            model_config=model_config,
            dataset_path=cfg.eval.dataset_path,
            checkpoint_path=cfg.eval.checkpoint_path,
            output_dir=cfg.eval.output_dir,
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.num_workers,
            transform_params=cfg.eval.transform_params,
            dataset_params=cfg.eval.dataset_params,
            device=cfg.eval.device,
        )

