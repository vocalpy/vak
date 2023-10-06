"""Unit tests for vak.prep.frame_classification.dataset_arrays"""
import json
import pathlib
import shutil

import crowsetta
import pytest

import vak.prep.frame_classification.dataset_arrays


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
    out = vak.prep.frame_classification.dataset_arrays.argsort_by_label_freq(annots)
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
def test_make_npy_files_for_each_split(config_type, model_name, audio_format, spect_format, annot_format,
                                       input_type, tmp_path, specific_dataset_df, specific_dataset_path):
    dataset_df = specific_dataset_df(config_type, model_name, annot_format, audio_format, spect_format)
    dataset_path = specific_dataset_path(config_type, model_name, annot_format, audio_format, spect_format)
    tmp_path_data_dir = tmp_path / 'data_dir'
    tmp_path_data_dir.mkdir()
    copy_dataset_df_files_to_tmp_path_data_dir(dataset_df, dataset_path, config_type, input_type, tmp_path_data_dir)

    tmp_dataset_path = tmp_path / 'dataset_dir'
    tmp_dataset_path.mkdir()

    if config_type != 'predict':
        with (dataset_path / 'labelmap.json').open('r') as fp:
            labelmap = json.load(fp)
    else:
        labelmap = None

    purpose = config_type

    vak.prep.frame_classification.dataset_arrays.make_npy_files_for_each_split(
        dataset_df,
        tmp_dataset_path,
        input_type,
        purpose,
        labelmap,
        audio_format,
    )

    splits = [
        split
        for split in sorted(dataset_df.split.dropna().unique())
        if split != "None"
    ]

    for split in splits:
        split_subdir = tmp_dataset_path / split
        if split != 'None':
            assert split_subdir.exists()
        elif split == 'None':
            assert not split_subdir.exists()

        split_df = dataset_df[dataset_df.split == split].copy()

        if purpose != "predict":
            annots = vak.common.annotation.from_df(split_df)
        else:
            annots = None

        if input_type == "audio":
            source_paths = split_df["audio_path"].values
        elif input_type == "spect":
            source_paths = split_df["spect_path"].values

        source_paths = [pathlib.Path(source_path) for source_path in source_paths]

        if annots:
            source_path_annot_tups = [
                (source_path, annot)
                for source_path, annot in zip(source_paths, annots)
            ]
        else:
            source_path_annot_tups = [
                (source_path, None)
                for source_path in source_paths
            ]

        for source_path_annot_tup in source_path_annot_tups:
            source_path, annot = source_path_annot_tup
            frames_array_file_that_should_exist = split_subdir / (
                source_path.stem
                + vak.datasets.frame_classification.constants.FRAMES_ARRAY_EXT
            )
            assert frames_array_file_that_should_exist.exists()
            if annot:
                frame_labels_file_that_should_exist = split_subdir / (
                    source_path.stem
                    + vak.datasets.frame_classification.constants.FRAME_LABELS_EXT
                )
                assert frame_labels_file_that_should_exist.exists()

        sample_id_vec_path = (
            split_subdir /
            vak.datasets.frame_classification.constants.SAMPLE_IDS_ARRAY_FILENAME
        )
        assert sample_id_vec_path.exists()

        inds_in_sample_vec_path = (
            split_subdir /
            vak.datasets.frame_classification.constants.INDS_IN_SAMPLE_ARRAY_FILENAME
        )
        assert inds_in_sample_vec_path.exists()
