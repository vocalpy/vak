import numpy as np
import pytest

import vak.transforms
import vak.transforms.functional
import vak.common.validators
from vak.datapipes.frame_classification import Metadata


class TestFramesStandardizer:

    @pytest.mark.parametrize(
        'mean_freqs, std_freqs, non_zero_std',
        [
            (
                np.array([0., 0., 0., 0., 0.]),
                np.array([1., 1., 1., 1., 1.]),
                np.array([0, 1, 2, 3, 4]),
            )
        ]
    )
    def test_instance(self, mean_freqs, std_freqs, non_zero_std):
        standardizer = vak.transforms.FramesStandardizer(
            mean_freqs=mean_freqs, std_freqs=std_freqs, non_zero_std=non_zero_std
        )
        assert isinstance(standardizer, vak.transforms.FramesStandardizer)
        for attr_name, expected in zip(
                ('mean_freqs', 'std_freqs', 'non_zero_std',),
                (mean_freqs, std_freqs, non_zero_std),
        ):
            assert hasattr(standardizer, attr_name)
            attr = getattr(standardizer, attr_name)
            assert isinstance(attr, np.ndarray)
            assert np.all(
                np.equal(attr, expected)
            )

        T_BINS = 100
        spect = np.random.rand(mean_freqs.shape[0], T_BINS)
        spect_out = standardizer(spect.copy())
        expected = vak.transforms.functional.standardize_spect(spect, mean_freqs, std_freqs, non_zero_std)
        assert np.all(
            np.equal(spect_out, expected)
        )

    def test_fit(self):
        spect = np.random.rand(513, 1000)
        standardizer = vak.transforms.FramesStandardizer.fit(spect)

        expected_mean_freqs = np.mean(spect, axis=1)
        expected_std_freqs = np.std(spect, axis=1)
        expected_non_zero_std = np.argwhere(expected_std_freqs != 0)
        # we convert to 1d vector in __init__
        expected_non_zero_std = vak.common.validators.column_or_1d(expected_non_zero_std)

        for attr_name, expected in zip(
                ('mean_freqs', 'std_freqs', 'non_zero_std',),
                (expected_mean_freqs, expected_std_freqs, expected_non_zero_std),
        ):
            assert hasattr(standardizer, attr_name)
            attr = getattr(standardizer, attr_name)
            assert isinstance(attr, np.ndarray)
            assert np.all(
                np.equal(attr, expected)
            )

    @pytest.mark.parametrize(
        'split',
        [
            'train',
            'val',
            None
        ]
    )
    def test_fit_inputs_targets_csv_path(self, split, train_cbin_notmat_df, specific_dataset_path):
        # we need dataset_path since paths in df are relative to it
        dataset_path = specific_dataset_path(
            config_type="train",
            model="TweetyNet",
            audio_format="cbin",
            annot_format="notmat"
        )
        metadata = Metadata.from_dataset_path(dataset_path)
        dataset_csv_path = dataset_path / metadata.dataset_csv_filename

        if split is None:
            split_to_test = 'train'
        else:
            split_to_test = split
        # ---- set up
        df_split = train_cbin_notmat_df[train_cbin_notmat_df.split == split_to_test].copy()
        spect_paths = df_split['frames_path'].values
        spect = vak.common.files.spect.load(dataset_path / spect_paths[0])[vak.common.constants.SPECT_KEY]
        mean_freqs = np.mean(spect, axis=1)
        std_freqs = np.std(spect, axis=1)

        for spect_path in spect_paths[1:]:
            spect = vak.common.files.spect.load(dataset_path / spect_path)[vak.common.constants.SPECT_KEY]
            mean_freqs += np.mean(spect, axis=1)
            std_freqs += np.std(spect, axis=1)
        expected_mean_freqs = mean_freqs / len(spect_paths)
        expected_std_freqs = std_freqs / len(spect_paths)
        expected_non_zero_std = np.argwhere(expected_std_freqs != 0)
        # we convert to 1d vector in __init__
        expected_non_zero_std = vak.common.validators.column_or_1d(expected_non_zero_std)

        # ---- actually do fit
        if split:
            standardizer = vak.transforms.FramesStandardizer.fit_inputs_targets_csv_path(dataset_csv_path, dataset_path, split=split)
        else:
            # this tests that default value for split 'train' works as expected
            standardizer = vak.transforms.FramesStandardizer.fit_inputs_targets_csv_path(dataset_csv_path, dataset_path)

        # ---- test
        for attr_name, expected in zip(
                ('mean_freqs', 'std_freqs', 'non_zero_std',),
                (expected_mean_freqs, expected_std_freqs, expected_non_zero_std),
        ):
            assert hasattr(standardizer, attr_name)
            attr = getattr(standardizer, attr_name)
            assert isinstance(attr, np.ndarray)
            assert np.all(
                np.equal(attr, expected)
            )

    @pytest.mark.parametrize(
        'split',
        [
            'train',
            'val',
            None
        ]
    )
    def test_fit_dataset_path(self, split, train_cbin_notmat_df, specific_dataset_path):
        # we need dataset_path since paths in df are relative to it
        dataset_path = specific_dataset_path(
            config_type="train",
            model="TweetyNet",
            audio_format="cbin",
            annot_format="notmat"
        )

        if split is None:
            split_to_test = 'train'
        else:
            split_to_test = split
        # ---- set up
        df_split = train_cbin_notmat_df[train_cbin_notmat_df.split == split_to_test].copy()
        spect_paths = df_split['frames_path'].values
        spect = vak.common.files.spect.load(dataset_path / spect_paths[0])[vak.common.constants.SPECT_KEY]
        mean_freqs = np.mean(spect, axis=1)
        std_freqs = np.std(spect, axis=1)

        for spect_path in spect_paths[1:]:
            spect = vak.common.files.spect.load(dataset_path / spect_path)[vak.common.constants.SPECT_KEY]
            mean_freqs += np.mean(spect, axis=1)
            std_freqs += np.std(spect, axis=1)
        expected_mean_freqs = mean_freqs / len(spect_paths)
        expected_std_freqs = std_freqs / len(spect_paths)
        expected_non_zero_std = np.argwhere(expected_std_freqs != 0)
        # we convert to 1d vector in __init__
        expected_non_zero_std = vak.common.validators.column_or_1d(expected_non_zero_std)

        # ---- actually do fit
        if split:
            standardizer = vak.transforms.FramesStandardizer.fit_dataset_path(dataset_path, split=split)
        else:
            # this tests that default value for split 'train' works as expected
            standardizer = vak.transforms.FramesStandardizer.fit_dataset_path(dataset_path)

        # ---- test
        for attr_name, expected in zip(
                ('mean_freqs', 'std_freqs', 'non_zero_std',),
                (expected_mean_freqs, expected_std_freqs, expected_non_zero_std),
        ):
            assert hasattr(standardizer, attr_name)
            attr = getattr(standardizer, attr_name)
            assert isinstance(attr, np.ndarray)
            assert np.all(
                np.equal(attr, expected)
            )
