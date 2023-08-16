"""Unit tests for vak.prep.frame_classification.dataset_arrays"""
import json

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


def copy_dataset_df_files_to_tmp_path_data_dir(dataset_df, dataset_path, tmp_path_data_dir):
    """Copy all the files in a dataset DataFrame to a `tmp_path_data_dir`,
    and change the paths in the Dataframe, so that we can then call
    `vak.prep.frame_classification.helper.move_files_into_split_subdirs`."""
    # TODO: rewrite to handle case where 'source' files of dataset are audio
    for paths_col in ('spect_path', 'annot_path'):
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
        ('train', 'TeenyTweetyNet', 'cbin', None, 'notmat', 'spect'),
        ('train', 'TeenyTweetyNet', None, 'mat', 'yarden', 'spect'),
        ('learncurve', 'TeenyTweetyNet', 'cbin', None, 'notmat', 'spect'),
    ]
)
def test_make_npy_files_for_each_split(config_type, model_name, audio_format, spect_format, annot_format,
                                       input_type, tmp_path, specific_dataset_df, specific_dataset_path):
    dataset_df = specific_dataset_df(config_type, model_name, annot_format, audio_format, spect_format)
    dataset_path = specific_dataset_path(config_type, model_name, annot_format, audio_format, spect_format)
    tmp_path_data_dir = tmp_path / 'data_dir'
    tmp_path_data_dir.mkdir()
    copy_dataset_df_files_to_tmp_path_data_dir(dataset_df, dataset_path, tmp_path_data_dir)

    tmp_dataset_path = tmp_path / 'dataset_dir'
    tmp_dataset_path.mkdir()

    with (dataset_path / 'labelmap.json').open('r') as fp:
        labelmap = json.load(fp)

    purpose = config_type

    vak.prep.frame_classification.dataset_arrays.make_npy_files_for_each_split(
        dataset_df,
        tmp_dataset_path,
        input_type,
        purpose,
        labelmap,
        audio_format,
    )

    for row in dataset_df[not dataset_df['split'].isna()]:
        split = row.split
        split_subdir = tmp_dataset_path / split
        if split != 'None':
            assert split_subdir.exists()
        elif split == 'None':
            assert not split_subdir.exists()

        # TODO: fix below so it works with predict, as in commented out code
        for col_name in (
                vak.datasets.frame_classification.constants.FRAMES_NPY_PATH_COL_NAME,
                vak.datasets.frame_classification.constants.FRAME_LABELS_NPY_PATH_COL_NAME,
        ):
            array_file_that_should_exist = getattr(row, col_name)
            expected_array_path = split_subdir / array_file_that_should_exist
            assert expected_array_path.exists()

        # if purpose != 'predict':
        #     for file_that_should_exist in (
        #             vak.datasets.frame_classification.FRAME_LABELS_ARRAY_FILENAME,
        #             vak.datasets.frame_classification.ANNOTATION_CSV_FILENAME,
        #     ):
        #         expected_path = split_subdir / file_that_should_exist
        #         assert expected_path.exists()
