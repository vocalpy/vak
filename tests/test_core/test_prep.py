"""tests for vak.core.prep module"""
from pathlib import Path
import shutil

import pandas as pd
from pandas.testing import assert_series_equal
import pytest

import vak.config
import vak.constants
import vak.core.train
import vak.paths
import vak.io.spect


# written as separate function so we can re-use in tests/unit/test_cli/test_prep.py
def prep_output_matches_expected(csv_path, df_returned_by_prep):
    assert Path(csv_path).exists()
    df_from_csv_path = pd.read_csv(csv_path)

    for column in vak.io.spect.DF_COLUMNS:
        if column == "duration":
            check_exact = False
        else:
            check_exact = True
        assert_series_equal(
            df_from_csv_path[column],
            df_returned_by_prep[column],
            check_exact=check_exact,
        )

    return True


@pytest.mark.parametrize(
    "config_type, audio_format, spect_format, annot_format",
    [
        ("eval", "cbin", None, "notmat"),
        ("learncurve", "cbin", None, "notmat"),
        ("predict", "cbin", None, "notmat"),
        ("predict", "wav", None, "birdsong-recognition-dataset"),
        ("train", "cbin", None, "notmat"),
        ("train", "wav", None, "birdsong-recognition-dataset"),
        ("train", None, "mat", "yarden"),
    ],
)
def test_prep(
    config_type,
    audio_format,
    spect_format,
    annot_format,
    specific_config,
    default_model,
    tmp_path,
):
    output_dir = tmp_path.joinpath(
        f"test_prep_{config_type}_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = {
        "section": "PREP",
        "option": "output_dir",
        "value": str(output_dir),
    }
    toml_path = specific_config(
        config_type=config_type,
        model=default_model,
        audio_format=audio_format,
        annot_format=annot_format,
        spect_format=spect_format,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    purpose = config_type.lower()
    vak_df, csv_path = vak.core.prep(
        data_dir=cfg.prep.data_dir,
        purpose=purpose,
        audio_format=cfg.prep.audio_format,
        spect_format=cfg.prep.spect_format,
        spect_output_dir=cfg.prep.spect_output_dir,
        spect_params=cfg.spect_params,
        annot_format=cfg.prep.annot_format,
        annot_file=cfg.prep.annot_file,
        labelset=cfg.prep.labelset,
        output_dir=cfg.prep.output_dir,
        train_dur=cfg.prep.train_dur,
        val_dur=cfg.prep.val_dur,
        test_dur=cfg.prep.test_dur,
    )

    assert prep_output_matches_expected(csv_path, vak_df)


def test_prep_with_single_audio_and_annot(source_test_data_root,
                                          specific_config,
                                          default_model,
                                          tmp_path):
    """
    regression test, checks that we avoid a repeat of
    https://github.com/NickleDave/vak/issues/467
    """
    data_dir = tmp_path / 'data_dir_with_single_audio_and_annot'
    data_dir.mkdir()
    source_data_dir = source_test_data_root / 'audio_cbin_annot_notmat/gy6or6/032412'
    cbins = sorted(source_data_dir.glob('*.cbin'))
    a_cbin = cbins[0]
    shutil.copy(a_cbin, data_dir)
    a_rec = a_notmat = a_cbin.parent / (a_cbin.stem + '.rec')
    assert a_rec.exists()
    shutil.copy(a_rec, data_dir)
    a_notmat = a_cbin.parent / (a_cbin.name + '.not.mat')
    assert a_notmat.exists()
    shutil.copy(a_notmat, data_dir)

    output_dir = tmp_path.joinpath(
        f"test_prep_eval_single_audio_and_annot"
    )
    output_dir.mkdir()

    options_to_change = [
        {
            "section": "PREP",
            "option": "data_dir",
            "value": str(data_dir),
        },
        {
            "section": "PREP",
            "option": "output_dir",
            "value": str(output_dir),
        },
    ]

    toml_path = specific_config(
        config_type='eval',
        model=default_model,
        audio_format='cbin',
        annot_format='notmat',
        spect_format=None,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    purpose = 'eval'
    vak_df, csv_path = vak.core.prep(
        data_dir=cfg.prep.data_dir,
        purpose=purpose,
        audio_format=cfg.prep.audio_format,
        spect_format=cfg.prep.spect_format,
        spect_output_dir=cfg.prep.spect_output_dir,
        spect_params=cfg.spect_params,
        annot_format=cfg.prep.annot_format,
        annot_file=cfg.prep.annot_file,
        labelset=cfg.prep.labelset,
        output_dir=cfg.prep.output_dir,
        train_dur=cfg.prep.train_dur,
        val_dur=cfg.prep.val_dur,
        test_dur=cfg.prep.test_dur,
    )

    assert len(vak_df) == 1


def test_prep_when_annot_has_single_segment(source_test_data_root,
                                            specific_config,
                                            default_model,
                                            tmp_path):
    """
    regression test, checks that we avoid a repeat of
    https://github.com/NickleDave/vak/issues/466
    """
    data_dir = source_test_data_root / 'audio_cbin_annot_notmat' / 'gy6or6-song-edited-to-have-single-segment'

    output_dir = tmp_path.joinpath(
        f"test_prep_eval_annot_with_single_segment"
    )
    output_dir.mkdir()

    options_to_change = [
        {
            "section": "PREP",
            "option": "data_dir",
            "value": str(data_dir),
        },
        {
            "section": "PREP",
            "option": "output_dir",
            "value": str(output_dir),
        },
    ]

    toml_path = specific_config(
        config_type='eval',
        model=default_model,
        audio_format='cbin',
        annot_format='notmat',
        spect_format=None,
        options_to_change=options_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    purpose = 'eval'
    vak_df, csv_path = vak.core.prep(
        data_dir=cfg.prep.data_dir,
        purpose=purpose,
        audio_format=cfg.prep.audio_format,
        spect_format=cfg.prep.spect_format,
        spect_output_dir=cfg.prep.spect_output_dir,
        spect_params=cfg.spect_params,
        annot_format=cfg.prep.annot_format,
        annot_file=cfg.prep.annot_file,
        labelset=cfg.prep.labelset,
        output_dir=cfg.prep.output_dir,
        train_dur=cfg.prep.train_dur,
        val_dur=cfg.prep.val_dur,
        test_dur=cfg.prep.test_dur,
    )

    assert len(vak_df) == 1
