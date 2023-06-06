import pathlib
import shutil

import pytest

import vak.prep.prep_helper


def copy_dataset_df_files_to_tmp_path_data_dir(dataset_df, dataset_path, tmp_path_data_dir):
    """Copy all the files in a dataset DataFrame to a `tmp_path_data_dir`,
    and change the paths in the Dataframe, so that we can then call
    `vak.prep.prep_helper.move_files_into_split_subdirs`."""
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
    'config_type, model_name, audio_format, spect_format, annot_format',
    [
        ('train', 'teenytweetynet', 'cbin', None, 'notmat'),
        ('train', 'teenytweetynet', None, 'mat', 'yarden'),
    ]
)
def test_move_files_into_split_subdirs(config_type, model_name, audio_format, spect_format, annot_format,
                                       tmp_path, specific_dataset_df, specific_dataset_path):
    dataset_df = specific_dataset_df(config_type, model_name, annot_format, audio_format, spect_format)
    dataset_path = specific_dataset_path(config_type, model_name, annot_format, audio_format, spect_format)
    tmp_path_data_dir = tmp_path / 'data_dir'
    tmp_path_data_dir.mkdir()
    copy_dataset_df_files_to_tmp_path_data_dir(dataset_df, dataset_path, tmp_path_data_dir)

    tmp_dataset_path = tmp_path / 'dataset_dir'
    tmp_dataset_path.mkdir()

    vak.prep.prep_helper.move_files_into_split_subdirs(dataset_df, tmp_dataset_path, purpose=config_type)

    for split in dataset_df['split'].dropna().unique():
        split_subdir = tmp_dataset_path / split
        assert split_subdir.exists()

        split_df = dataset_df[dataset_df['split'] == split]
        # for path_col in ('spect_path', 'annot_path'):
        spect_paths = split_df['spect_path'].values
        for spect_path in spect_paths:
            new_path = split_subdir / pathlib.Path(spect_path).name
            assert new_path.exists()

        if len(dataset_df["annot_path"].unique()) == 1:
            # --> there is a single annotation file associated with all rows
            # in this case we copy the single annotation file to the root of the dataset directory
            annot_path = dataset_df["annot_path"].unique().item()
            assert (tmp_dataset_path / annot_path).exists()
        elif len(dataset_df["annot_path"].unique()) == len(dataset_df):
            annot_paths = split_df['annot_path'].values
            for annot_path in annot_paths:
                new_path = split_subdir / pathlib.Path(annot_path).name
                assert new_path.exists()


@pytest.mark.parametrize(
    'data_dir_name, timestamp',
    [
        ('bird1', '230319_115852')
    ]
)
def test_get_dataset_csv_filename(data_dir_name, timestamp):
    out = vak.prep.prep_helper.get_dataset_csv_filename(data_dir_name, timestamp)
    assert isinstance(out, str)
    assert out.startswith(data_dir_name)
    assert out.endswith('.csv')
    out =  out.replace('.csv', '')
    assert out.endswith(timestamp)  # after removing extension
    assert '_prep_' in out


@pytest.mark.parametrize(
    'data_dir_name, timestamp',
    [
        ('bird1', '230319_115852')
    ]
)
def test_get_dataset_csv_path(data_dir_name, timestamp, tmp_path):
    out = vak.prep.prep_helper.get_dataset_csv_path(tmp_path, data_dir_name, timestamp)
    assert isinstance(out, pathlib.Path)
    assert out.name == vak.prep.prep_helper.get_dataset_csv_filename(data_dir_name, timestamp)
    assert out.parent == tmp_path


def test_add_split_col(audio_dir_cbin,
                       default_spect_params,
                       labelset_notmat,
                       tmp_path):
    """test that ``add_split_col`` adds a 'split' column
    to a DataFrame, where all values in the Series are the
    specified split (a string)"""
    vak_df = vak.prep.spectrogram_dataset.prep.prep_spectrogram_dataset(
        data_dir=audio_dir_cbin,
        labelset=labelset_notmat,
        annot_format="notmat",
        audio_format="cbin",
        spect_output_dir=tmp_path,
        spect_format=None,
        annot_file=None,
        spect_params=default_spect_params,
    )

    assert "split" not in vak_df.columns

    vak_df = vak.prep.prep_helper.add_split_col(vak_df, split="train")
    assert "split" in vak_df.columns

    assert vak_df["split"].unique().item() == "train"
