from unittest import mock

import pandas as pd
import pytest

import vak

FAKE_SOURCE_FILES_DF = pd.DataFrame.from_records(
    [
        {'audio_path': 'bird0-2023.10.12.cbin',
         'spect_path': 'bird0-2023.10.12.cbin.spect.npz',
         'annot_path': 'bird0-2023.10.12.cbin.not.mat'}
    ]
)


@pytest.mark.parametrize(
    'config_type, model_name, audio_format, spect_format, annot_format, input_type',
    [
        ('train', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        ('predict', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        ('eval', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        ('train', 'TweetyNet', None, 'mat', 'yarden', 'spect'),
        ('learncurve', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
    ]
)
def test_get_or_make_source_files(
        config_type, model_name, audio_format, spect_format, annot_format,
        input_type, tmp_path, specific_config_toml_path
):
    """Test that this `vak.prep.frame_classification.get_or_make_source_files` dispatches correctly.

    Other unit tests already test the functions that this function calls.
    """
    toml_path = specific_config_toml_path(
        config_type,
        model_name,
        annot_format,
        audio_format,
        spect_format,
    )

    cfg = vak.config.Config.from_toml_path(toml_path)

    # ---- set up ----
    tmp_dataset_path = tmp_path / 'dataset_dir'
    tmp_dataset_path.mkdir()

    if cfg.prep.input_type == 'audio':
        with mock.patch('vak.prep.frame_classification.source_files.prep_audio_dataset', autospec=True) as mock_prep_audio_dataset:
            mock_prep_audio_dataset.return_value = FAKE_SOURCE_FILES_DF

            out: pd.DataFrame = vak.prep.frame_classification.get_or_make_source_files(
                cfg.prep.data_dir,
                cfg.prep.input_type,
                cfg.prep.audio_format,
                cfg.prep.spect_format,
                cfg.prep.spect_params,
                tmp_dataset_path,
                cfg.prep.annot_format,
                cfg.prep.annot_file,
                cfg.prep.labelset,
                cfg.prep.audio_dask_bag_kwargs,
            )

            assert mock_prep_audio_dataset.called
            assert isinstance(out, pd.DataFrame)

    elif cfg.prep.input_type == 'spect':
        with mock.patch(
                'vak.prep.frame_classification.source_files.prep_spectrogram_dataset', autospec=True
        ) as mock_prep_spect_dataset:
            mock_prep_spect_dataset.return_value = FAKE_SOURCE_FILES_DF

            out: pd.DataFrame = vak.prep.frame_classification.get_or_make_source_files(
                cfg.prep.data_dir,
                cfg.prep.input_type,
                cfg.prep.audio_format,
                cfg.prep.spect_format,
                cfg.prep.spect_params,
                tmp_dataset_path,
                cfg.prep.annot_format,
                cfg.prep.annot_file,
                cfg.prep.labelset,
                cfg.prep.audio_dask_bag_kwargs,
            )
            assert mock_prep_spect_dataset.called
            assert isinstance(out, pd.DataFrame)
