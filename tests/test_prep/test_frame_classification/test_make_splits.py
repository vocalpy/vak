"""Unit tests for vak.prep.frame_classification.make_splits"""
import json
import pathlib
import shutil

import crowsetta
import numpy as np
import pandas as pd
import pytest

import vak.prep.frame_classification.make_splits


@pytest.mark.parametrize(
    'annots, expected_sort_inds',
    [
        (
            [
                crowsetta.Annotation(seq=crowsetta.Sequence.from_keyword(
                    onsets_s=[0.1, 0.3, 0.5], offsets_s=[0.2, 0.4, 0.6], labels=['a', 'b', 'b']
                ), annot_path='./fake'),
                crowsetta.Annotation(seq=crowsetta.Sequence.from_keyword(
                    onsets_s=[0.1, 0.3, 0.5], offsets_s=[0.2, 0.4, 0.6], labels=['b', 'b', 'b']
                ), annot_path='./fake'),
            ],
            [0, 1,]
        ),
        (
            [
                crowsetta.Annotation(seq=crowsetta.Sequence.from_keyword(
                    onsets_s=[0.1, 0.3, 0.5], offsets_s=[0.2, 0.4, 0.6], labels=['a', 'b', 'b']
                ), annot_path='./fake'),
                crowsetta.Annotation(seq=crowsetta.Sequence.from_keyword(
                    onsets_s=[0.1, 0.3, 0.5], offsets_s=[0.2, 0.4, 0.6], labels=['b', 'b', 'b']
                ), annot_path='./fake'),
                crowsetta.Annotation(seq=crowsetta.Sequence.from_keyword(
                    onsets_s=[0.1, 0.3, 0.5], offsets_s=[0.2, 0.4, 0.6], labels=['b', 'b', 'b']
                ), annot_path='./fake'),
            ],
            [0, 1, 2],
        ),
    ]
)
def test_argsort_by_label_freq(annots, expected_sort_inds):
    out = vak.prep.frame_classification.make_splits.argsort_by_label_freq(annots)
    assert isinstance(out, list)
    assert out == expected_sort_inds


def copy_dataset_df_files_to_tmp_path_data_dir(dataset_df, dataset_path, config_type, input_type, tmp_path_data_dir):
    """Copy all the files in a dataset DataFrame to a `tmp_path_data_dir`,
    and change the paths in the Dataframe, so that we can then call
    `vak.prep.frame_classification.helper.move_files_into_split_subdirs`."""
    paths_cols = []
    if input_type == 'spect':
        paths_cols.append('spect_path')
    elif input_type == 'audio':
        paths_cols.append('audio_path')
    if config_type != 'predict':
        paths_cols.append('annot_path')
    for paths_col in paths_cols:
        paths = dataset_df[paths_col].values
        new_paths = []
        for path in paths:
            new_path = shutil.copy(src=dataset_path / path, dst=tmp_path_data_dir)
            new_paths.append(new_path)
        dataset_df[paths_col] = new_paths
    return dataset_df


@pytest.mark.parametrize(
    'config_type, model_name, audio_format, spect_format, annot_format, input_type',
    [
        ('train', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        ('predict', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        ('eval', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        ('train', 'TweetyNet', None, 'mat', 'yarden', 'spect'),
        ('learncurve', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        # TODO: add audio cases
    ]
)
def test_make_splits(config_type, model_name, audio_format, spect_format, annot_format,
                     input_type, tmp_path, specific_config_toml_path, specific_source_files_with_split_df):
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

    purpose = config_type

    dataset_df = specific_source_files_with_split_df(
        config_type,
        model_name,
        annot_format,
        audio_format,
        spect_format,
    )
    if purpose != "predict":
        # TODO: add option to generate predict using existing dataset, so we can get labelmap from it
        map_unlabeled_segments = vak.prep.sequence_dataset.has_unlabeled_segments(
            dataset_df
        )
        labelmap = vak.common.labels.to_map(
            cfg.prep.labelset, map_background=map_unlabeled_segments
        )
    else:
        labelmap = None

    dataset_df_with_splits = vak.prep.frame_classification.make_splits.make_splits(
        dataset_df,
        tmp_dataset_path,
        cfg.prep.input_type,
        purpose,
        labelmap,
        cfg.prep.audio_format,
    )
    assert isinstance(dataset_df_with_splits, pd.DataFrame)

    splits = [
        split
        for split in sorted(dataset_df_with_splits.split.dropna().unique())
        if split != "None"
    ]

    for split in splits:
        split_subdir = tmp_dataset_path / split
        if split != 'None':
            assert split_subdir.exists()
        elif split == 'None':
            assert not split_subdir.exists()

        split_df = dataset_df_with_splits[
            dataset_df_with_splits.split == split
        ].copy()

        assert vak.datapipes.frame_classification.constants.FRAMES_PATH_COL_NAME in split_df.columns

        frames_paths = split_df[
            vak.datapipes.frame_classification.constants.FRAMES_PATH_COL_NAME
        ].values

        if purpose != "predict":
            assert vak.datapipes.frame_classification.constants.MULTI_FRAME_LABELS_PATH_COL_NAME in split_df.columns

            frame_labels_paths = split_df[
                vak.datapipes.frame_classification.constants.MULTI_FRAME_LABELS_PATH_COL_NAME
            ].values

            annots = vak.common.annotation.from_df(split_df)

            frames_tuples = [
                (frames_path, frame_labels_path, annot)
                for frames_path, frame_labels_path, annot in zip(
                    frames_paths, frame_labels_paths, annots
                )
            ]
        else:
            frames_tuples = [
                (frames_path, None, None)
                for frames_path in frames_paths
            ]

        sample_id_vecs, inds_in_sample_vecs = [], []
        for sample_id, frames_tuple in enumerate(frames_tuples):
            frames_path, frame_labels_path, annot = frames_tuple
            frames_file_that_should_exist = tmp_dataset_path / frames_path
            assert frames_file_that_should_exist.exists()

            # NOTE we load frames to confirm we can and also to make indexing vectors we use to test,
            # see next code block
            frames = vak.datapipes.frame_classification.helper.load_frames(tmp_dataset_path / frames_path, input_type)
            assert isinstance(frames, np.ndarray)

            # make indexing vectors that we use to test
            n_frames = frames.shape[-1]
            sample_id_vecs.append(np.ones((n_frames,)).astype(np.int32) * sample_id)
            inds_in_sample_vecs.append(np.arange(n_frames))

            if frame_labels_path is not None and annot is not None:
                frame_labels_file_that_should_exist = tmp_dataset_path / frame_labels_path
                assert frame_labels_file_that_should_exist.exists()

                if input_type == "audio":
                    _, samplefreq = vak.common.constants.AUDIO_FORMAT_FUNC_MAP[
                        audio_format
                    ](tmp_dataset_path / frames_path)
                    frame_times = np.arange(frames.shape[-1]) / samplefreq
                elif input_type == "spect":
                    spect_dict = vak.common.files.spect.load(tmp_dataset_path / frames_path, "npz")
                    frame_times = spect_dict[vak.common.constants.TIMEBINS_KEY]

                lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
                expected_frame_labels = vak.transforms.frame_labels.from_segments(
                    lbls_int,
                    annot.seq.onsets_s,
                    annot.seq.offsets_s,
                    frame_times,
                    background_label=labelmap[vak.common.constants.DEFAULT_BACKGROUND_LABEL],
                )
                frame_labels = np.load(frame_labels_file_that_should_exist)
                assert np.array_equal(frame_labels, expected_frame_labels)

        # assert there are no remaining .spect.npz files in dataset path (root)
        # because they were moved in to splits, and we removed any remaining that were not put into splits
        spect_npz_files_not_in_split = sorted(
            tmp_dataset_path.glob(f'*{vak.common.constants.SPECT_NPZ_EXTENSION}')
        )
        assert len(spect_npz_files_not_in_split) == 0

        sample_id_vec_path = (
            split_subdir /
            vak.datapipes.frame_classification.constants.SAMPLE_IDS_ARRAY_FILENAME
        )
        assert sample_id_vec_path.exists()

        expected_sample_id_vec = np.concatenate(sample_id_vecs)
        sample_id_vec = np.load(sample_id_vec_path)
        assert np.array_equal(sample_id_vec, expected_sample_id_vec)

        inds_in_sample_vec_path = (
            split_subdir /
            vak.datapipes.frame_classification.constants.INDS_IN_SAMPLE_ARRAY_FILENAME
        )
        assert inds_in_sample_vec_path.exists()

        expected_inds_in_sample_vec = np.concatenate(inds_in_sample_vecs)
        inds_in_sample_vec = np.load(inds_in_sample_vec_path)
        assert np.array_equal(inds_in_sample_vec, expected_inds_in_sample_vec)
