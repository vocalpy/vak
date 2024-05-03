"""tests for vak.config.prep module"""
import copy

import pytest

import vak.config.prep


class TestPrepConfig:
    @pytest.mark.parametrize(
        'config_dict',
        [
            {
                'annot_format': 'notmat',
                'audio_format': 'cbin',
                'data_dir': './tests/data_for_tests/source/audio_cbin_annot_notmat/gy6or6/032412',
                'dataset_type': 'parametric umap',
                'input_type': 'spect',
                'labelset': 'iabcdefghjk',
                'output_dir': './tests/data_for_tests/generated/prep/eval/audio_cbin_annot_notmat/ConvEncoderUMAP',
                'spect_params': {'fft_size': 512,
                                'step_size': 32,
                                'transform_type': 'log_spect_plus_one'},
                'test_dur': 0.2
            },
            {
                'annot_format': 'notmat',
                'audio_format': 'cbin',
                'data_dir': './tests/data_for_tests/source/audio_cbin_annot_notmat/gy6or6/032312',
                'dataset_type': 'frame classification',
                'input_type': 'spect',
                'labelset': 'iabcdefghjk',
                'output_dir': './tests/data_for_tests/generated/prep/train/audio_cbin_annot_notmat/TweetyNet',
                'spect_params': {'fft_size': 512,
                                'freq_cutoffs': [500, 10000],
                                'step_size': 64,
                                'thresh': 6.25,
                                'transform_type': 'log_spect'},
                'test_dur': 30,
                'train_dur': 50,
                'val_dur': 15
            },
        ]
    )
    def test_init(self, config_dict):
        config_dict['spect_params'] = vak.config.SpectParamsConfig(**config_dict['spect_params'])

        prep_config = vak.config.PrepConfig(**config_dict)

        assert isinstance(prep_config, vak.config.prep.PrepConfig)
        for key, val in config_dict.items():
            assert hasattr(prep_config, key)
            if key == 'data_dir' or key == 'output_dir':
                assert getattr(prep_config, key) == vak.common.converters.expanded_user_path(val)
            elif key == 'labelset':
                assert getattr(prep_config, key) == vak.common.converters.labelset_to_set(val)
            else:
                assert getattr(prep_config, key) == val

    @pytest.mark.parametrize(
        'config_dict',
            [
                {
                    'annot_format': 'notmat',
                    'audio_format': 'cbin',
                    'spect_format': 'mat',
                    'data_dir': './tests/data_for_tests/source/audio_cbin_annot_notmat/gy6or6/032412',
                    'dataset_type': 'parametric umap',
                    'input_type': 'spect',
                    'labelset': 'iabcdefghjk',
                    'output_dir': './tests/data_for_tests/generated/prep/eval/audio_cbin_annot_notmat/ConvEncoderUMAP',
                    'spect_params': {'fft_size': 512,
                                    'step_size': 32,
                                    'transform_type': 'log_spect_plus_one'},
                    'test_dur': 0.2
                },
        ]
    )
    def test_both_audio_and_spect_format_raises(
        self, config_dict,
    ):
        """test that a config with both an audio and a spect format raises a ValueError"""
        # need to do this set-up so we don't mask one error with another
        config_dict['spect_params'] = vak.config.SpectParamsConfig(**config_dict['spect_params'])

        with pytest.raises(ValueError):
            prep_config = vak.config.PrepConfig(**config_dict)

    @pytest.mark.parametrize(
        'config_dict',
            [
                {
                    'annot_format': 'notmat',
                    'data_dir': './tests/data_for_tests/source/audio_cbin_annot_notmat/gy6or6/032412',
                    'dataset_type': 'parametric umap',
                    'input_type': 'spect',
                    'labelset': 'iabcdefghjk',
                    'output_dir': './tests/data_for_tests/generated/prep/eval/audio_cbin_annot_notmat/ConvEncoderUMAP',
                    'spect_params': {'fft_size': 512,
                                    'step_size': 32,
                                    'transform_type': 'log_spect_plus_one'},
                    'test_dur': 0.2
                },
        ]
    )
    def test_neither_audio_nor_spect_format_raises(
        self, config_dict
    ):
        """test that a config without either an audio or a spect format raises a ValueError"""
        # need to do this set-up so we don't mask one error with another
        config_dict['spect_params'] = vak.config.SpectParamsConfig(**config_dict['spect_params'])

        with pytest.raises(ValueError):
            prep_config = vak.config.PrepConfig(**config_dict)

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'annot_format': 'notmat',
                    'audio_format': 'cbin',
                    'data_dir': './tests/data_for_tests/source/audio_cbin_annot_notmat/gy6or6/032412',
                    'dataset_type': 'parametric umap',
                    'input_type': 'spect',
                    'labelset': 'iabcdefghjk',
                    'output_dir': './tests/data_for_tests/generated/prep/eval/audio_cbin_annot_notmat/ConvEncoderUMAP',
                    'spect_params': {'fft_size': 512,
                                    'step_size': 32,
                                    'transform_type': 'log_spect_plus_one'},
                    'test_dur': 0.2
                },
                {
                    'annot_format': 'notmat',
                    'audio_format': 'cbin',
                    'data_dir': './tests/data_for_tests/source/audio_cbin_annot_notmat/gy6or6/032312',
                    'dataset_type': 'frame classification',
                    'input_type': 'spect',
                    'labelset': 'iabcdefghjk',
                    'output_dir': './tests/data_for_tests/generated/prep/train/audio_cbin_annot_notmat/TweetyNet',
                    'spect_params': {'fft_size': 512,
                                    'freq_cutoffs': [500, 10000],
                                    'step_size': 64,
                                    'thresh': 6.25,
                                    'transform_type': 'log_spect'},
                    'test_dur': 30,
                    'train_dur': 50,
                    'val_dur': 15
                },
            ]
    )
    def test_from_config_dict(self, config_dict):
        # we have to make a copy since `from_config_dict` mutates the dict
        config_dict_copy = copy.deepcopy(config_dict)

        prep_config = vak.config.prep.PrepConfig.from_config_dict(config_dict_copy)

        assert isinstance(prep_config, vak.config.prep.PrepConfig)
        for key, val in config_dict.items():
            assert hasattr(prep_config, key)
            if key == 'data_dir' or key == 'output_dir':
                assert getattr(prep_config, key) == vak.common.converters.expanded_user_path(val)
            elif key == 'labelset':
                assert getattr(prep_config, key) == vak.common.converters.labelset_to_set(val)
            elif key == 'spect_params':
                assert getattr(prep_config, key) == vak.config.SpectParamsConfig(**val)
            else:
                assert getattr(prep_config, key) == val

    def test_from_config_dict_real_config(
            self, a_generated_config_dict
    ):
        prep_config = vak.config.prep.PrepConfig.from_config_dict(a_generated_config_dict['prep'])
        assert isinstance(prep_config, vak.config.prep.PrepConfig)
