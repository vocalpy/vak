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

    spect_output_dir = tmp_path.joinpath(
        f"spectrograms_test_prep_{config_type}_{audio_format}_{spect_format}_{annot_format}"
    )
    spect_output_dir.mkdir()

    options_to_change = [
        {
            "section": "PREP",
            "option": "output_dir",
            "value": str(output_dir),
        },
        {
            "section": "PREP",
            "option": "spect_output_dir",
            "value": str(spect_output_dir),
        },
    ]
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


@pytest.mark.parametrize(
    "config_type, audio_format, spect_format, annot_format",
    [
        ("eval", "cbin", None, "notmat"),
        ("learncurve", "cbin", None, "notmat"),
        ("train", "cbin", None, "notmat"),
        ("train", "wav", None, "birdsong-recognition-dataset"),
        ("train", None, "mat", "yarden"),
    ],
)
def test_prep_raises_when_labelset_required_but_is_none(
    config_type,
    audio_format,
    spect_format,
    annot_format,
    specific_config,
    default_model,
    tmp_path,
):
    """Test that `prep` raises a ValueError when the config
    requires a `labelset`,
    i.e., is one of {'train','learncurve', 'eval'},
    but it is left as None.

    Regression test for https://github.com/vocalpy/vak/issues/468.
    """
    output_dir = tmp_path.joinpath(
        f"test_prep_{config_type}_{audio_format}_{spect_format}_{annot_format}"
    )
    output_dir.mkdir()

    options_to_change = [
        {"section": "PREP",
         "option": "output_dir",
         "value": str(output_dir),
         },
        {"section": "PREP",
         "option": "labelset",
         "value": "DELETE-OPTION",
         },
    ]
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
    with pytest.raises(ValueError):
        vak.core.prep(
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
        {
            "section": "PREP",
            "option": "spect_output_dir",
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
        {
            "section": "PREP",
            "option": "spect_output_dir",
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


@pytest.mark.parametrize(
    "dir_option_to_change",
    [
        {"section": "PREP", "option": "data_dir", "value": '/obviously/does/not/exist/data'},
        {"section": "PREP", "option": "output_dir", "value": '/obviously/does/not/exist/output'},
        {"section": "PREP", "option": "spect_output_dir", "value": '/obviously/does/not/exist/spect_output'},
    ],
)
def test_prep_raises_not_a_directory(
    dir_option_to_change,
    specific_config,
    default_model,
    tmp_path,
):
    """Test that `core.prep` raise NotADirectory error
    when one of the following is not a directory:
    data_dir, output_dir, spect_output_dir
    """
    toml_path = specific_config(
        config_type="train",
        model="teenytweetynet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=dir_option_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    purpose = "train"
    with pytest.raises(NotADirectoryError):
        vak.core.prep(
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


@pytest.mark.parametrize(
    "path_option_to_change",
    [
        {"section": "PREP", "option": "annot_file", "value": '/obviously/does/not/exist/annot.mat'},
    ],
)
def test_prep_raises_file_not_found(
    path_option_to_change,
    specific_config,
    default_model,
    tmp_path,
):
    """Test that `core.prep` raise FileNotFound error
    when one of the following does not exist:
    annot_file

    Structuring unit test this way in case other path
    parameters get added.
    """
    toml_path = specific_config(
        config_type="train",
        model="teenytweetynet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=path_option_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    purpose = "train"
    with pytest.raises(FileNotFoundError):
        vak.core.prep(
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
