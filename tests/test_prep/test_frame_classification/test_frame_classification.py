"""Tests for vak.prep.frame_classification.frame_classification.prep_frame_classification_dataset"""
import json
import pathlib
import shutil

import pandas as pd
from pandas.testing import assert_series_equal
import pytest

import vak


def assert_prep_output_matches_expected(dataset_path, df_returned_by_prep):
    dataset_path = pathlib.Path(dataset_path)
    assert dataset_path.exists()
    assert dataset_path.is_dir()

    log_path = sorted(dataset_path.glob('*log'))
    assert len(log_path) == 1

    meta_json_path = dataset_path / vak.datasets.frame_classification.Metadata.METADATA_JSON_FILENAME
    assert meta_json_path.exists()

    with meta_json_path.open('r') as fp:
        meta_json = json.load(fp)

    dataset_csv_path = dataset_path / meta_json['dataset_csv_filename']
    assert dataset_csv_path.exists()

    df_from_dataset_path = pd.read_csv(dataset_csv_path)

    for column in vak.prep.spectrogram_dataset.spect_helper.DF_COLUMNS:
        if column == "duration":
            check_exact = False
        else:
            check_exact = True
        try:
            assert_series_equal(
                df_from_dataset_path[column],
                df_returned_by_prep[column],
                check_exact=check_exact,
            )
        except:
            breakpoint()

    for column in ('spect_path', 'annot_path'):
        paths = df_from_dataset_path[column].values
        if not all([isinstance(path, str) for path in paths]):
            continue
        for path in paths:
            path = pathlib.Path(path)
            assert (dataset_path / path).exists()


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
def test_prep_frame_classification_dataset(
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

    options_to_change = [
        {
            "section": "PREP",
            "option": "output_dir",
            "value": str(output_dir),
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
    dataset_df, dataset_path = vak.prep.frame_classification.frame_classification.prep_frame_classification_dataset(
        data_dir=cfg.prep.data_dir,
        input_type=cfg.prep.input_type,
        purpose=purpose,
        audio_format=cfg.prep.audio_format,
        spect_format=cfg.prep.spect_format,
        spect_params=cfg.spect_params,
        annot_format=cfg.prep.annot_format,
        annot_file=cfg.prep.annot_file,
        labelset=cfg.prep.labelset,
        output_dir=cfg.prep.output_dir,
        train_dur=cfg.prep.train_dur,
        val_dur=cfg.prep.val_dur,
        test_dur=cfg.prep.test_dur,
        train_set_durs=cfg.prep.train_set_durs,
        num_replicates=cfg.prep.num_replicates,
    )

    assert_prep_output_matches_expected(dataset_path, dataset_df)


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
def test_prep_frame_classification_dataset_raises_when_labelset_required_but_is_none(
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
        vak.prep.frame_classification.frame_classification.prep_frame_classification_dataset(
            data_dir=cfg.prep.data_dir,
            input_type=cfg.prep.input_type,
            purpose=purpose,
            audio_format=cfg.prep.audio_format,
            spect_format=cfg.prep.spect_format,
            spect_params=cfg.spect_params,
            annot_format=cfg.prep.annot_format,
            annot_file=cfg.prep.annot_file,
            labelset=cfg.prep.labelset,
            output_dir=cfg.prep.output_dir,
            train_dur=cfg.prep.train_dur,
            val_dur=cfg.prep.val_dur,
            test_dur=cfg.prep.test_dur,
        )


def test_prep_frame_classification_dataset_with_single_audio_and_annot(source_test_data_root,
                                          specific_config,
                                          default_model,
                                          tmp_path):
    """
    regression test, checks that we avoid a repeat of
    https://github.com/vocalpy/vak/issues/467
    """
    data_dir = tmp_path / 'data_dir_with_single_audio_and_annot'
    data_dir.mkdir()
    source_data_dir = source_test_data_root / 'audio_cbin_annot_notmat/gy6or6/032412'
    cbins = sorted(source_data_dir.glob('*.cbin'))
    a_cbin = cbins[0]
    shutil.copy(a_cbin, data_dir)
    a_rec = a_cbin.parent / (a_cbin.stem + '.rec')
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
    dataset_df, dataset_path = vak.prep.frame_classification.frame_classification.prep_frame_classification_dataset(
        data_dir=cfg.prep.data_dir,
        input_type=cfg.prep.input_type,
        purpose=purpose,
        audio_format=cfg.prep.audio_format,
        spect_format=cfg.prep.spect_format,
        spect_params=cfg.spect_params,
        annot_format=cfg.prep.annot_format,
        annot_file=cfg.prep.annot_file,
        labelset=cfg.prep.labelset,
        output_dir=cfg.prep.output_dir,
        train_dur=cfg.prep.train_dur,
        val_dur=cfg.prep.val_dur,
        test_dur=cfg.prep.test_dur,
    )

    assert len(dataset_df) == 1


def test_prep_frame_classification_dataset_when_annot_has_single_segment(source_test_data_root,
                                                                         specific_config,
                                                                         default_model,
                                                                         tmp_path):
    """
    regression test, checks that we avoid a repeat of
    https://github.com/vocalpy/vak/issues/466
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
    dataset_df, dataset_path = vak.prep.frame_classification.frame_classification.prep_frame_classification_dataset(
        data_dir=cfg.prep.data_dir,
        input_type=cfg.prep.input_type,
        purpose=purpose,
        audio_format=cfg.prep.audio_format,
        spect_format=cfg.prep.spect_format,
        spect_params=cfg.spect_params,
        annot_format=cfg.prep.annot_format,
        annot_file=cfg.prep.annot_file,
        labelset=cfg.prep.labelset,
        output_dir=cfg.prep.output_dir,
        train_dur=cfg.prep.train_dur,
        val_dur=cfg.prep.val_dur,
        test_dur=cfg.prep.test_dur,
    )

    assert len(dataset_df) == 1


@pytest.mark.parametrize(
    "dir_option_to_change",
    [
        {"section": "PREP", "option": "data_dir", "value": '/obviously/does/not/exist/data'},
        {"section": "PREP", "option": "output_dir", "value": '/obviously/does/not/exist/output'},
    ],
)
def test_prep_frame_classification_dataset_raises_not_a_directory(
    dir_option_to_change,
    specific_config,
    default_model,
    tmp_path,
):
    """Test that `core.prep` raise NotADirectory error
    when one of the following is not a directory:
    data_dir, output_dir
    """
    toml_path = specific_config(
        config_type="train",
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=dir_option_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    purpose = "train"
    with pytest.raises(NotADirectoryError):
        vak.prep.frame_classification.frame_classification.prep_frame_classification_dataset(
            data_dir=cfg.prep.data_dir,
            input_type=cfg.prep.input_type,
            purpose=purpose,
            audio_format=cfg.prep.audio_format,
            spect_format=cfg.prep.spect_format,
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
def test_prep_frame_classification_dataset_raises_file_not_found(
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
        model="TweetyNet",
        audio_format="cbin",
        annot_format="notmat",
        spect_format=None,
        options_to_change=path_option_to_change,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    purpose = "train"
    with pytest.raises(FileNotFoundError):
        vak.prep.frame_classification.frame_classification.prep_frame_classification_dataset(
            data_dir=cfg.prep.data_dir,
            input_type=cfg.prep.input_type,
            purpose=purpose,
            audio_format=cfg.prep.audio_format,
            spect_format=cfg.prep.spect_format,
            spect_params=cfg.spect_params,
            annot_format=cfg.prep.annot_format,
            annot_file=cfg.prep.annot_file,
            labelset=cfg.prep.labelset,
            output_dir=cfg.prep.output_dir,
            train_dur=cfg.prep.train_dur,
            val_dur=cfg.prep.val_dur,
            test_dur=cfg.prep.test_dur,
        )
