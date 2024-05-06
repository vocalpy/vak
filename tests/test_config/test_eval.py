"""tests for vak.config.eval module"""
import pytest

import vak.config


class TestEval:

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'checkpoint_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/TweetyNet/checkpoints/max-val-acc-checkpoint.pt',
                    'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/labelmap.json',
                    'batch_size': 11,
                    'num_workers': 16,
                    'trainer': {'accelerator': 'gpu', 'devices': [0]},
                    'frames_standardizer_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/StandardizeSpect',
                    'output_dir': './tests/data_for_tests/generated/results/eval/audio_cbin_annot_notmat/TweetyNet',
                    'post_tfm_kwargs': {
                        'majority_vote': True, 'min_segment_dur': 0.02
                        },
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

        eval_config = vak.config.EvalConfig(**config_dict)

        assert isinstance(eval_config, vak.config.EvalConfig)

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'checkpoint_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/TweetyNet/checkpoints/max-val-acc-checkpoint.pt',
                    'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/labelmap.json',
                    'batch_size': 11,
                    'num_workers': 16,
                    'trainer': {'accelerator': 'gpu', 'devices': [0]},
                    'frames_standardizer_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/StandardizeSpect',
                    'output_dir': './tests/data_for_tests/generated/results/eval/audio_cbin_annot_notmat/TweetyNet',
                    'post_tfm_kwargs': {
                        'majority_vote': True, 'min_segment_dur': 0.02
                        },
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
        eval_config = vak.config.EvalConfig.from_config_dict(config_dict)

        assert isinstance(eval_config, vak.config.EvalConfig)

    def test_from_config_dict_with_real_config(self, a_generated_eval_config_dict):
        eval_table = a_generated_eval_config_dict["eval"]

        eval_config = vak.config.eval.EvalConfig.from_config_dict(eval_table)

        assert isinstance(eval_config, vak.config.eval.EvalConfig)

    @pytest.mark.parametrize(
        'config_dict, expected_exception',
        [
            # missing 'model', should raise KeyError
            (
                {
                    'checkpoint_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/TweetyNet/checkpoints/max-val-acc-checkpoint.pt',
                    'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/labelmap.json',
                    'batch_size': 11,
                    'num_workers': 16,
                    'trainer': {'accelerator': 'gpu', 'devices': [0]},
                    'frames_standardizer_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/StandardizeSpect',
                    'output_dir': './tests/data_for_tests/generated/results/eval/audio_cbin_annot_notmat/TweetyNet',
                    'post_tfm_kwargs': {
                        'majority_vote': True, 'min_segment_dur': 0.02
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
                    'checkpoint_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/TweetyNet/checkpoints/max-val-acc-checkpoint.pt',
                    'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/labelmap.json',
                    'batch_size': 11,
                    'num_workers': 16,
                    'trainer': {'accelerator': 'gpu', 'devices': [0]},
                    'frames_standardizer_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/StandardizeSpect',
                    'output_dir': './tests/data_for_tests/generated/results/eval/audio_cbin_annot_notmat/TweetyNet',
                    'post_tfm_kwargs': {
                        'majority_vote': True, 'min_segment_dur': 0.02
                        },
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
            # missing 'checkpoint_path', should raise KeyError
            (
                {
                    'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/labelmap.json',
                    'batch_size': 11,
                    'num_workers': 16,
                    'trainer': {'accelerator': 'gpu', 'devices': [0]},
                    'frames_standardizer_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/StandardizeSpect',
                    'output_dir': './tests/data_for_tests/generated/results/eval/audio_cbin_annot_notmat/TweetyNet',
                    'post_tfm_kwargs': {
                        'majority_vote': True, 'min_segment_dur': 0.02
                        },
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
            # missing 'output_dir', should raise KeyError
            (
                {
                    'checkpoint_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/TweetyNet/checkpoints/max-val-acc-checkpoint.pt',
                    'labelmap_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/labelmap.json',
                    'batch_size': 11,
                    'num_workers': 16,
                    'trainer': {'accelerator': 'gpu', 'devices': [0]},
                    'frames_standardizer_path': '~/Documents/repos/coding/birdsong/TweetyNet/results/BFSongRepository/gy6or6/results_200620_165308/StandardizeSpect',
                    'post_tfm_kwargs': {
                        'majority_vote': True, 'min_segment_dur': 0.02
                        },
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
            )
        ]
    )
    def test_from_config_dict_raises(self, config_dict, expected_exception):
        with pytest.raises(expected_exception):
            vak.config.EvalConfig.from_config_dict(config_dict)