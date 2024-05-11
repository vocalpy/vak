"""tests for vak.config.predict module"""
import pytest

import vak.config.predict


class TestPredictConfig:

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'frames_standardizer_path': '/home/user/results_181014_194418/spect_scaler',
                    'checkpoint_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/bl26lb16/results_200620_164245/TweetyNet/checkpoints/max-val-acc-checkpoint.pt',
                    'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/bl26lb16/results_200620_164245/labelmap.json',
                    'batch_size': 11,
                    'num_workers': 16,
                    'trainer': {'accelerator': 'gpu', 'devices': [0]},
                    'output_dir': './tests/data_for_tests/generated/results/predict/audio_cbin_annot_notmat/TweetyNet',
                    'annot_csv_filename': 'bl26lb16.041912.annot.csv',
                    'model': {
                        'TweetyNet': {
                            'network': {
                                'conv1_filters': 8,
                                'conv1_kernel_size': [3, 3],
                                'conv2_filters': 16,
                                'conv2_kernel_size': [5, 5],
                                'pool1_size': [4, 1],
                                'pool1_stride': [4, 1],
                                'pool2_size': [4, 1],
                                'pool2_stride': [4, 1],
                                'hidden_size': 32
                                },
                                'optimizer': {'lr': 0.001}
                            }
                        },
                        'dataset': {
                            'path': '~/some/path/I/made/up/for/now'
                        },
                    }
            ]
    )
    def test_init(self, config_dict):
        config_dict['model'] = vak.config.ModelConfig.from_config_dict(config_dict['model'])
        config_dict['dataset'] = vak.config.DatasetConfig.from_config_dict(config_dict['dataset'])
        config_dict['trainer'] = vak.config.TrainerConfig(**config_dict['trainer'])

        predict_config = vak.config.PredictConfig(**config_dict)

        assert isinstance(predict_config, vak.config.PredictConfig)

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'frames_standardizer_path': '/home/user/results_181014_194418/spect_scaler',
                    'checkpoint_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/bl26lb16/results_200620_164245/TweetyNet/checkpoints/max-val-acc-checkpoint.pt',
                    'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/bl26lb16/results_200620_164245/labelmap.json',
                    'batch_size': 11,
                    'num_workers': 16,
                    'trainer': {'accelerator': 'gpu', 'devices': [0]},
                    'output_dir': './tests/data_for_tests/generated/results/predict/audio_cbin_annot_notmat/TweetyNet',
                    'annot_csv_filename': 'bl26lb16.041912.annot.csv',
                    'model': {
                        'TweetyNet': {
                            'network': {
                                'conv1_filters': 8,
                                'conv1_kernel_size': [3, 3],
                                'conv2_filters': 16,
                                'conv2_kernel_size': [5, 5],
                                'pool1_size': [4, 1],
                                'pool1_stride': [4, 1],
                                'pool2_size': [4, 1],
                                'pool2_stride': [4, 1],
                                'hidden_size': 32
                                },
                                'optimizer': {'lr': 0.001}
                            }
                        },
                        'dataset': {
                            'path': '~/some/path/I/made/up/for/now'
                        },
                    }
            ]
    )
    def test_from_config_dict(self, config_dict):
        predict_config = vak.config.PredictConfig.from_config_dict(config_dict)

        assert isinstance(predict_config, vak.config.PredictConfig)

    def test_from_config_dict_with_real_config(self, a_generated_predict_config_dict):
        predict_table = a_generated_predict_config_dict["predict"]

        predict_config = vak.config.predict.PredictConfig.from_config_dict(predict_table)

        assert isinstance(predict_config, vak.config.predict.PredictConfig)

    @pytest.mark.parametrize(
            'config_dict, expected_exception',
            [
                # missing 'checkpoint_path', should raise KeyError
                (
                    {
                        'frames_standardizer_path': '/home/user/results_181014_194418/spect_scaler',
                        'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/bl26lb16/results_200620_164245/labelmap.json',
                        'batch_size': 11,
                        'num_workers': 16,
                        'trainer': {'accelerator': 'gpu', 'devices': [0]},
                        'output_dir': './tests/data_for_tests/generated/results/predict/audio_cbin_annot_notmat/TweetyNet',
                        'annot_csv_filename': 'bl26lb16.041912.annot.csv',
                        'model': {
                            'TweetyNet': {
                                'network': {
                                    'conv1_filters': 8,
                                    'conv1_kernel_size': [3, 3],
                                    'conv2_filters': 16,
                                    'conv2_kernel_size': [5, 5],
                                    'pool1_size': [4, 1],
                                    'pool1_stride': [4, 1],
                                    'pool2_size': [4, 1],
                                    'pool2_stride': [4, 1],
                                    'hidden_size': 32
                                    },
                                    'optimizer': {'lr': 0.001}
                                }
                            },
                        'dataset': {
                            'path': '~/some/path/I/made/up/for/now'
                        },
                    },
                    KeyError
                ),
                # missing 'dataset', should raise KeyError
                (
                    {
                        'frames_standardizer_path': '/home/user/results_181014_194418/spect_scaler',
                        'checkpoint_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/bl26lb16/results_200620_164245/TweetyNet/checkpoints/max-val-acc-checkpoint.pt',
                        'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/bl26lb16/results_200620_164245/labelmap.json',
                        'batch_size': 11,
                        'num_workers': 16,
                        'trainer': {'accelerator': 'gpu', 'devices': [0]},
                        'output_dir': './tests/data_for_tests/generated/results/predict/audio_cbin_annot_notmat/TweetyNet',
                        'annot_csv_filename': 'bl26lb16.041912.annot.csv',
                        'model': {
                            'TweetyNet': {
                                'network': {
                                    'conv1_filters': 8,
                                    'conv1_kernel_size': [3, 3],
                                    'conv2_filters': 16,
                                    'conv2_kernel_size': [5, 5],
                                    'pool1_size': [4, 1],
                                    'pool1_stride': [4, 1],
                                    'pool2_size': [4, 1],
                                    'pool2_stride': [4, 1],
                                    'hidden_size': 32
                                    },
                                    'optimizer': {'lr': 0.001}
                                }
                            },
                    },
                    KeyError
                ),
                # missing 'model', should raise KeyError
                (
                    {
                        'frames_standardizer_path': '/home/user/results_181014_194418/spect_scaler',
                        'checkpoint_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/bl26lb16/results_200620_164245/TweetyNet/checkpoints/max-val-acc-checkpoint.pt',
                        'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/bl26lb16/results_200620_164245/labelmap.json',
                        'batch_size': 11,
                        'num_workers': 16,
                        'trainer': {'accelerator': 'gpu', 'devices': [0]},
                        'output_dir': './tests/data_for_tests/generated/results/predict/audio_cbin_annot_notmat/TweetyNet',
                        'annot_csv_filename': 'bl26lb16.041912.annot.csv',
                        'dataset': {
                            'path': '~/some/path/I/made/up/for/now'
                        },
                    },
                    KeyError
                ),
            ]
    )
    def test_from_config_dict_raises(self, config_dict, expected_exception):
        with pytest.raises(expected_exception):
            vak.config.PredictConfig.from_config_dict(config_dict)
