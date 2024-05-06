"""tests for vak.config.learncurve module"""
import pytest

import vak.config.learncurve


class TestLearncurveConfig:

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'standardize_frames': True,
                    'batch_size': 11,
                    'num_epochs': 2,
                    'val_step': 50,
                    'ckpt_step': 200,
                    'patience': 4,
                    'num_workers': 16,
                    'trainer': {'accelerator': 'gpu', 'devices': [0]},
                    'root_results_dir': './tests/data_for_tests/generated/results/train/audio_cbin_annot_notmat/TweetyNet',
                    'post_tfm_kwargs': {'majority_vote': True, 'min_segment_dur': 0.02},
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
                            'optimizer': {
                                'lr': 0.001
                            }
                        }
                    },
                    'dataset': {
                        'path': 'tests/data_for_tests/generated/prep/train/audio_cbin_annot_notmat/TweetyNet/032312-vak-frame-classification-dataset-generated-240502_234819'
                    }
                }
            ]
    )
    def test_init(self, config_dict):
        config_dict['model'] = vak.config.ModelConfig.from_config_dict(config_dict['model'])
        config_dict['dataset'] = vak.config.DatasetConfig.from_config_dict(config_dict['dataset'])
        config_dict['trainer'] = vak.config.TrainerConfig(**config_dict['trainer'])

        learncurve_config = vak.config.LearncurveConfig(**config_dict)

        assert isinstance(learncurve_config, vak.config.LearncurveConfig)

    @pytest.mark.parametrize(
            'config_dict',
            [
                {
                    'standardize_frames': True,
                    'batch_size': 11,
                    'num_epochs': 2,
                    'val_step': 50,
                    'ckpt_step': 200,
                    'patience': 4,
                    'num_workers': 16,
                    'trainer': {'accelerator': 'gpu', 'devices': [0]},
                    'root_results_dir': './tests/data_for_tests/generated/results/train/audio_cbin_annot_notmat/TweetyNet',
                    'post_tfm_kwargs': {'majority_vote': True, 'min_segment_dur': 0.02},
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
                            'optimizer': {
                                'lr': 0.001
                            }
                        }
                    },
                    'dataset': {
                        'path': 'tests/data_for_tests/generated/prep/train/audio_cbin_annot_notmat/TweetyNet/032312-vak-frame-classification-dataset-generated-240502_234819'
                    }
                }
            ]
    )
    def test_from_config_dict(self, config_dict):
        learncurve_config = vak.config.LearncurveConfig.from_config_dict(config_dict)

        assert isinstance(learncurve_config, vak.config.LearncurveConfig)

    def test_from_config_dict_with_real_config(self, a_generated_learncurve_config_dict):
        """test that instantiating LearncurveConfig class works as expected"""
        learncurve_table = a_generated_learncurve_config_dict["learncurve"]

        config = vak.config.learncurve.LearncurveConfig.from_config_dict(
            learncurve_table
        )

        assert isinstance(config, vak.config.learncurve.LearncurveConfig)

    @pytest.mark.parametrize(
            'config_dict, expected_exception',
            [
                # missing 'model', should raise KeyError
                (
                    {
                        'standardize_frames': True,
                        'batch_size': 11,
                        'num_epochs': 2,
                        'val_step': 50,
                        'ckpt_step': 200,
                        'patience': 4,
                        'num_workers': 16,
                        'trainer': {'accelerator': 'gpu', 'devices': [0]},
                        'root_results_dir': './tests/data_for_tests/generated/results/train/audio_cbin_annot_notmat/TweetyNet',
                        'post_tfm_kwargs': {'majority_vote': True, 'min_segment_dur': 0.02},
                        'dataset': {
                            'path': 'tests/data_for_tests/generated/prep/train/audio_cbin_annot_notmat/TweetyNet/032312-vak-frame-classification-dataset-generated-240502_234819'
                        }
                    },
                    KeyError
                ),
                # missing 'dataset', should raise KeyError
               (
                    {
                        'standardize_frames': True,
                        'batch_size': 11,
                        'num_epochs': 2,
                        'val_step': 50,
                        'ckpt_step': 200,
                        'patience': 4,
                        'num_workers': 16,
                        'trainer': {'accelerator': 'gpu', 'devices': [0]},
                        'root_results_dir': './tests/data_for_tests/generated/results/train/audio_cbin_annot_notmat/TweetyNet',
                        'post_tfm_kwargs': {'majority_vote': True, 'min_segment_dur': 0.02},
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
                                'optimizer': {
                                    'lr': 0.001
                                }
                            }
                        },
                    },
                    KeyError
                ),
                # missing 'root_results_dir', should raise KeyError
                (
                    {
                        'standardize_frames': True,
                        'batch_size': 11,
                        'num_epochs': 2,
                        'val_step': 50,
                        'ckpt_step': 200,
                        'patience': 4,
                        'num_workers': 16,
                        'trainer': {'accelerator': 'gpu', 'devices': [0]},
                        'post_tfm_kwargs': {'majority_vote': True, 'min_segment_dur': 0.02},
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
                                'optimizer': {
                                    'lr': 0.001
                                }
                            }
                        },
                        'dataset': {
                            'path': 'tests/data_for_tests/generated/prep/train/audio_cbin_annot_notmat/TweetyNet/032312-vak-frame-classification-dataset-generated-240502_234819'
                        }
                    },
                    KeyError
                )
            ]
    )
    def test_from_config_dict_raises(self, config_dict, expected_exception):
        with pytest.raises(expected_exception):
            vak.config.LearncurveConfig.from_config_dict(config_dict)
