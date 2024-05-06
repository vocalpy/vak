import json
import shutil
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import vak.common.converters
import vak.common.labels
import vak.common.paths
import vak.prep.frame_classification

@pytest.mark.parametrize(
    'model_name, audio_format, annot_format, input_type',
    [
        ('TweetyNet', 'cbin', 'notmat', 'spect')
    ]
)
def test_make_index_vectors_for_each_subsets(
    model_name, audio_format, annot_format, input_type, specific_config_toml_path, trainer_table, tmp_path,
):
    root_results_dir = tmp_path.joinpath("tmp_root_results_dir")
    root_results_dir.mkdir()
    keys_to_change = [
        {
            "table": "learncurve",
            "key": "root_results_dir",
            "value": str(root_results_dir),
        },
    ]
    toml_path = specific_config_toml_path(
        config_type="learncurve",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)

    dataset_path = cfg.learncurve.dataset.path
    metadata = vak.datapipes.frame_classification.Metadata.from_dataset_path(dataset_path)
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)

    subsets_df = dataset_df[
        ~dataset_df['subset'].isnull()
    ]

    tmp_dataset_path = tmp_path / f"test_make_learncurve_splits_from_dataset_df"
    shutil.copytree(dataset_path, tmp_dataset_path)
    # delete all the subset indices vectors, since we're about to test that we make them
    for train_dur in cfg.prep.train_set_durs:
        for replicate_num in range(1, cfg.prep.num_replicates + 1):
            train_dur_replicate_subset_name = vak.common.learncurve.get_train_dur_replicate_subset_name(
                    train_dur, replicate_num
                )
            sample_id_vec_path = (tmp_dataset_path / "train" /
                                  vak.datapipes.frame_classification.helper.sample_ids_array_filename_for_subset(
                                      train_dur_replicate_subset_name)
                                  )
            sample_id_vec_path.unlink()
            inds_in_sample_vec_path = (tmp_dataset_path / "train" /
                                       vak.datapipes.frame_classification.helper.inds_in_sample_array_filename_for_subset(
                                       train_dur_replicate_subset_name)
                                  )
            inds_in_sample_vec_path.unlink()

    vak.prep.frame_classification.learncurve.make_index_vectors_for_each_subset(
        subsets_df,
        tmp_dataset_path,
        input_type,
    )

    assert sorted(subsets_df['train_dur'].unique()) == cfg.prep.train_set_durs
    assert sorted(
        subsets_df['replicate_num'].unique()
    ) == list(range(1, cfg.prep.num_replicates + 1))

    # assert that each expected split name is in data frame
    for train_dur in cfg.prep.train_set_durs:
        train_dur_df = subsets_df[np.isclose(subsets_df['train_dur'], train_dur)].copy()
        # assert correct number of replicates for this train duration
        assert sorted(
            train_dur_df['replicate_num']
        ) == list(range(1, cfg.prep.num_replicates + 1))

        for replicate_num in range(1, cfg.prep.num_replicates + 1):
            subset_name = vak.common.learncurve.get_train_dur_replicate_subset_name(
                    train_dur, replicate_num
                )

            # test that indexing vectors got made
            sample_id_vec_path = (tmp_dataset_path / "train" /
                                  vak.datapipes.frame_classification.helper.sample_ids_array_filename_for_subset(
                                      subset_name)
                                  )
            assert sample_id_vec_path.exists()

            inds_in_sample_vec_path = (tmp_dataset_path / "train" /
                                       vak.datapipes.frame_classification.helper.inds_in_sample_array_filename_for_subset(
                                           subset_name)
                                       )
            assert inds_in_sample_vec_path.exists()

            this_subset_df = subsets_df[subsets_df['subset'] == subset_name]
            frames_paths = this_subset_df[
                vak.datapipes.frame_classification.constants.FRAMES_PATH_COL_NAME
            ].values
            sample_id_vec, inds_in_sample_vec = [], []
            for sample_id, frames_path in enumerate(frames_paths):
                # make indexing vectors that we use to test
                frames = vak.datapipes.frame_classification.helper.load_frames(tmp_dataset_path / frames_path,
                                                                              input_type)
                n_frames = frames.shape[-1]
                sample_id_vec.append(np.ones((n_frames,)).astype(np.int32) * sample_id)
                inds_in_sample_vec.append(np.arange(n_frames))
            expected_sample_id_vec = np.concatenate(sample_id_vec)
            expected_inds_in_sample_vec = np.concatenate(inds_in_sample_vec)
            sample_id_vec = np.load(sample_id_vec_path)
            assert np.array_equal(sample_id_vec, expected_sample_id_vec)
            inds_in_sample_vec = np.load(inds_in_sample_vec_path)
            assert np.array_equal(inds_in_sample_vec, expected_inds_in_sample_vec)


@pytest.mark.parametrize(
    'model_name, audio_format, annot_format, input_type',
    [
        ('TweetyNet', 'cbin', 'notmat', 'spect')
    ]
)
def test_make_subsets_from_dataset_df(
    model_name, audio_format, annot_format, input_type, specific_config_toml_path, trainer_table, tmp_path,
):
    root_results_dir = tmp_path.joinpath("tmp_root_results_dir")
    root_results_dir.mkdir()
    keys_to_change = [
        {
            "table": "learncurve",
            "key": "root_results_dir",
            "value": str(root_results_dir),
        },
    ]
    toml_path = specific_config_toml_path(
        config_type="learncurve",
        model=model_name,
        audio_format=audio_format,
        annot_format=annot_format,
        keys_to_change=keys_to_change,
    )
    cfg = vak.config.Config.from_toml_path(toml_path)

    dataset_path = cfg.learncurve.dataset.path
    metadata = vak.datapipes.frame_classification.Metadata.from_dataset_path(dataset_path)
    dataset_csv_path = dataset_path / metadata.dataset_csv_filename
    dataset_df = pd.read_csv(dataset_csv_path)

    labelmap_path = dataset_path / "labelmap.json"
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)

    tmp_dataset_path = tmp_path / f"test_make_learncurve_splits_from_dataset_df"
    shutil.copytree(dataset_path, tmp_dataset_path)
    # delete all the subset indices vectors, since we're about to test that we make them
    for train_dur in cfg.prep.train_set_durs:
        for replicate_num in range(1, cfg.prep.num_replicates + 1):
            train_dur_replicate_subset_name = vak.common.learncurve.get_train_dur_replicate_subset_name(
                    train_dur, replicate_num
                )
            sample_id_vec_path = (tmp_dataset_path / "train" /
                                  vak.datapipes.frame_classification.helper.sample_ids_array_filename_for_subset(
                                      train_dur_replicate_subset_name)
                                  )
            sample_id_vec_path.unlink()
            inds_in_sample_vec_path = (tmp_dataset_path / "train" /
                                       vak.datapipes.frame_classification.helper.inds_in_sample_array_filename_for_subset(
                                       train_dur_replicate_subset_name)
                                  )
            inds_in_sample_vec_path.unlink()

    # now reset the dataset df to what it would have been before we passed it into `make_splits`
    dataset_df = dataset_df[
        # drop any rows where there *is* a train dur -- because these are the subsets
        dataset_df['train_dur'].isnull()
        # drop the columns added by ``make_splits``, then reset the index
    ].drop(columns=['subset', 'train_dur', 'replicate_num']).reset_index(drop=True)

    with mock.patch('vak.prep.frame_classification.learncurve.make_index_vectors_for_each_subset') as mock_idx_vectors:
        out = vak.prep.frame_classification.learncurve.make_subsets_from_dataset_df(
            dataset_df,
            input_type,
            cfg.prep.train_set_durs,
            cfg.prep.num_replicates,
            tmp_dataset_path,
            labelmap,
        )
        assert mock_idx_vectors.called

    assert isinstance(out, pd.DataFrame)

    for added_column in ('subset', 'train_dur', 'replicate_num'):
        assert added_column in out.columns

    subsets_df = out[
        ~out['subset'].isnull()
    ]
    assert sorted(subsets_df['train_dur'].unique()) == cfg.prep.train_set_durs
    assert sorted(
        subsets_df['replicate_num'].unique()
    ) == list(range(1, cfg.prep.num_replicates + 1))

    # assert that each expected split name is in data frame
    for train_dur in cfg.prep.train_set_durs:
        train_dur_df = subsets_df[np.isclose(subsets_df['train_dur'], train_dur)].copy()
        # assert correct number of replicates for this train duration
        assert sorted(
            train_dur_df['replicate_num']
        ) == list(range(1, cfg.prep.num_replicates + 1))

        for replicate_num in range(1, cfg.prep.num_replicates + 1):
            subset_name = vak.common.learncurve.get_train_dur_replicate_subset_name(
                    train_dur, replicate_num
                )

            # assert this train_dur + replicate split exists in dataframe
            assert np.isin(subset_name, subsets_df['subset'].values)
            this_subset_df = subsets_df[subsets_df['subset'] == subset_name]

            # assert that it has the correct duration
            assert this_subset_df['duration'].sum() >= train_dur
