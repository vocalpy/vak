"""tests for vak.io.dataframe module"""
from pathlib import Path

import pandas as pd

import vak.annotation
import vak.constants
import vak.io.audio
import vak.io.dataframe
import vak.io.spect


def returned_dataframe_matches_expected(
    df_returned,
    data_dir,
    labelset,
    annot_format,
    audio_format,
    spect_format,
    spect_output_dir=None,
    annot_file=None,
    expected_audio_paths=None,
    not_expected_audio_paths=None,
    spect_file_ext=".spect.npz",
    expected_spect_paths=None,
    not_expected_spect_paths=None,
):
    """tests that dataframe returned by ``vak.io.dataframe.from_files``
    matches expected dataframe."""
    assert isinstance(df_returned, pd.DataFrame)

    assert df_returned.columns.values.tolist() == vak.io.spect.DF_COLUMNS

    annot_format_from_df = df_returned.annot_format.unique()
    assert len(annot_format_from_df) == 1
    annot_format_from_df = annot_format_from_df[0]

    if annot_format is not None:
        # test format from dataframe is specified format
        assert annot_format_from_df == annot_format
        annot_list = vak.annotation.from_df(df_returned)
    else:
        # if no annotation format specified, check that `annot_format` is `none`
        assert annot_format_from_df == vak.constants.NO_ANNOTATION_FORMAT
        annot_list = None

    if annot_file:
        assert all(
            [Path(annot_path) == annot_file for annot_path in df_returned["annot_path"]]
        )

    if labelset:
        # test that filtering vocalizations by labelset worked
        assert annot_list is not None, "labelset specified but annot_list was None"
        for annot in annot_list:
            labels = annot.seq.labels
            labelset_from_labels = set(labels)
            # should be true that set of labels from each annotation is a subset of labelset
            assert labelset_from_labels.issubset(set(labelset))

    audio_files_from_df = [Path(audio_path) for audio_path in df_returned.audio_path]
    spect_paths_from_df = [Path(spect_path) for spect_path in df_returned.spect_path]

    # --- which assertions to run depends on whether we made the dataframe
    # from audio files or spect files ----
    if audio_format:  # implies that --> we made the dataframe from audio files

        # test that all audio files came from data_dir
        for audio_file_from_df in audio_files_from_df:
            assert Path(audio_file_from_df).parent == data_dir

        # test that each audio file has a corresponding spect file in `spect_path` column
        spect_file_names = [spect_path.name for spect_path in spect_paths_from_df]
        expected_spect_files = [
            source_audio_file.name + spect_file_ext
            for source_audio_file in expected_audio_paths
        ]
        assert all(
            [
                expected_spect_file in spect_file_names
                for expected_spect_file in expected_spect_files
            ]
        )

        # if there are audio files we expect to **not** be in audio_path
        # -- because the associated annotations have labels not in labelset --
        # then test those files are **not** in spect_path
        if not_expected_audio_paths is not None:
            not_expected_spect_files = [
                source_audio_file.name + spect_file_ext
                for source_audio_file in not_expected_audio_paths
            ]
            assert all(
                [
                    not_expected_spect_file not in spect_file_names
                    for not_expected_spect_file in not_expected_spect_files
                ]
            )

        # test that all the generated spectrogram files are in a
        # newly-created directory inside spect_output_dir
        assert all(
            [
                spect_path.parents[1] == spect_output_dir
                for spect_path in spect_paths_from_df
            ]
        )

    elif spect_format:  # implies that --> we made the dataframe from audio files
        for expected_spect_path in list(expected_spect_paths):
            assert expected_spect_path in spect_paths_from_df
            spect_paths_from_df.remove(expected_spect_path)

        # test that **only** expected paths were in DataFrame
        if not_expected_spect_paths is not None:
            for not_expected_spect_path in not_expected_spect_paths:
                assert not_expected_spect_path not in spect_paths_from_df

        # test that **only** expected paths were in DataFrame
        # spect_paths_from_df should be empty after popping off all the expected paths
        assert (
            len(spect_paths_from_df) == 0
        )  # yes I know this isn't "Pythonic". It's readable, go away.

    return True  # all asserts passed


def spect_files_have_correct_keys(df_returned, 
                                  spect_params):
    spect_paths_from_df = [Path(spect_path) for spect_path in df_returned.spect_path]
    for spect_path in spect_paths_from_df:
        spect_dict = vak.files.spect.load(spect_path)
        for key_type in ['freqbins_key', 'timebins_key', 'spect_key', 'audio_path_key']:
            if key_type in spect_params:
                key = spect_params[key_type]
            else:
                # if we didn't pass in this key type, don't check for it
                # this is for `audio_path` which is not strictly required currently
                continue
            assert key in spect_dict

    return True


def test_from_files_with_audio_cbin_with_labelset(
    audio_dir_cbin,
    default_spect_params,
    labelset_notmat,
    audio_list_cbin_all_labels_in_labelset,
    audio_list_cbin_labels_not_in_labelset,
    spect_list_npz_all_labels_in_labelset,
    spect_list_npz_labels_not_in_labelset,
    tmp_path,
):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .cbin audio files
    and specify an annotation format"""
    vak_df = vak.io.dataframe.from_files(
        data_dir=audio_dir_cbin,
        labelset=labelset_notmat,
        annot_format="notmat",
        audio_format="cbin",
        spect_output_dir=tmp_path,
        spect_format=None,
        annot_file=None,
        spect_params=default_spect_params,
    )

    assert returned_dataframe_matches_expected(
        vak_df,
        data_dir=audio_dir_cbin,
        labelset=labelset_notmat,
        annot_format="notmat",
        audio_format="cbin",
        spect_format=None,
        spect_output_dir=tmp_path,
        annot_file=None,
        expected_audio_paths=audio_list_cbin_all_labels_in_labelset,
        not_expected_audio_paths=audio_list_cbin_labels_not_in_labelset,
        expected_spect_paths=spect_list_npz_all_labels_in_labelset,
        not_expected_spect_paths=spect_list_npz_labels_not_in_labelset,
    )

    assert spect_files_have_correct_keys(vak_df, default_spect_params)


def test_from_files_with_audio_cbin_no_annot(
    audio_dir_cbin, default_spect_params, labelset_notmat, audio_list_cbin, tmp_path
):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .cbin audio files
    and  **do not** specify an annotation format"""
    vak_df = vak.io.dataframe.from_files(
        data_dir=audio_dir_cbin,
        annot_format=None,
        labelset=None,
        audio_format="cbin",
        spect_output_dir=tmp_path,
        spect_format=None,
        annot_file=None,
        spect_params=default_spect_params,
    )

    assert returned_dataframe_matches_expected(
        vak_df,
        data_dir=audio_dir_cbin,
        annot_format=None,
        labelset=None,
        audio_format="cbin",
        spect_format=None,
        spect_output_dir=tmp_path,
        expected_audio_paths=audio_list_cbin,
        annot_file=None,
    )

    assert spect_files_have_correct_keys(vak_df, default_spect_params)


def test_from_files_with_audio_cbin_no_labelset(
    audio_dir_cbin, default_spect_params, audio_list_cbin, tmp_path
):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .cbin audio files
    and specify an annotation format"""
    vak_df = vak.io.dataframe.from_files(
        data_dir=audio_dir_cbin,
        annot_format="notmat",
        labelset=None,
        audio_format="cbin",
        spect_format=None,
        spect_output_dir=tmp_path,
        annot_file=None,
        spect_params=default_spect_params,
    )

    assert returned_dataframe_matches_expected(
        vak_df,
        data_dir=audio_dir_cbin,
        annot_format="notmat",
        labelset=None,
        audio_format="cbin",
        spect_format=None,
        spect_output_dir=tmp_path,
        expected_audio_paths=audio_list_cbin,
        annot_file=None,
    )

    assert spect_files_have_correct_keys(vak_df, default_spect_params)


def test_from_files_with_audio_cbin_non_default_spect_file_keys(
    audio_dir_cbin,
    default_spect_params,
    labelset_notmat,
    audio_list_cbin_all_labels_in_labelset,
    audio_list_cbin_labels_not_in_labelset,
    spect_list_npz_all_labels_in_labelset,
    spect_list_npz_labels_not_in_labelset,
    tmp_path,
):
    """test that ``vak.io.dataframe.from_files`` works
    when we specify different keys for accessing
    arrays in array files
    """
    spect_params = {k:v for k, v in default_spect_params.items()}
    spect_params.update(
        dict(
            freqbins_key="freqbins",
            timebins_key="timebins",
            spect_key="spect",
            audio_path_key="audio_path"
        )
    )
    vak_df = vak.io.dataframe.from_files(
        data_dir=audio_dir_cbin,
        labelset=labelset_notmat,
        annot_format="notmat",
        audio_format="cbin",
        spect_output_dir=tmp_path,
        spect_format=None,
        annot_file=None,
        spect_params=spect_params,
    )

    assert returned_dataframe_matches_expected(
        vak_df,
        data_dir=audio_dir_cbin,
        labelset=labelset_notmat,
        annot_format="notmat",
        audio_format="cbin",
        spect_format=None,
        spect_output_dir=tmp_path,
        annot_file=None,
        expected_audio_paths=audio_list_cbin_all_labels_in_labelset,
        not_expected_audio_paths=audio_list_cbin_labels_not_in_labelset,
        expected_spect_paths=spect_list_npz_all_labels_in_labelset,
        not_expected_spect_paths=spect_list_npz_labels_not_in_labelset,
    )

    assert spect_files_have_correct_keys(vak_df, spect_params)


def test_from_files_with_spect_mat(
    spect_dir_mat,
    default_spect_params,
    labelset_yarden,
    annot_file_yarden,
    spect_list_mat_all_labels_in_labelset,
    spect_list_mat_labels_not_in_labelset,
):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .mat array files
    and specify an annotation format"""
    vak_df = vak.io.dataframe.from_files(
        data_dir=spect_dir_mat,
        labelset=labelset_yarden,
        annot_format="yarden",
        audio_format=None,
        spect_format="mat",
        annot_file=annot_file_yarden,
        spect_params=None,
    )

    assert returned_dataframe_matches_expected(
        vak_df,
        data_dir=spect_dir_mat,
        labelset=labelset_yarden,
        annot_format="yarden",
        audio_format=None,
        spect_format="mat",
        annot_file=annot_file_yarden,
        expected_spect_paths=spect_list_mat_all_labels_in_labelset,
        not_expected_spect_paths=spect_list_mat_labels_not_in_labelset,
    )

    del default_spect_params['audio_path_key']  # 'audio_path' not strictly required
    assert spect_files_have_correct_keys(vak_df, default_spect_params)


def test_from_files_with_spect_mat_no_annot(default_spect_params,
                                            spect_dir_mat,
                                            spect_list_mat):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .mat array files
    and **do not** specify an annotation format"""
    vak_df = vak.io.dataframe.from_files(
        data_dir=spect_dir_mat,
        labelset=None,
        annot_format=None,
        audio_format=None,
        spect_format="mat",
        annot_file=None,
        spect_params=None,
    )

    assert returned_dataframe_matches_expected(
        vak_df,
        data_dir=spect_dir_mat,
        labelset=None,
        annot_format=None,
        audio_format=None,
        spect_format="mat",
        annot_file=None,
        expected_spect_paths=spect_list_mat,
    )

    del default_spect_params['audio_path_key']  # 'audio_path' not strictly required
    assert spect_files_have_correct_keys(vak_df, default_spect_params)


def test_from_files_with_spect_mat_no_labelset(spect_dir_mat, 
                                               default_spect_params,
                                               labelset_yarden,
                                               annot_file_yarden,
                                               annot_list_yarden,
                                               spect_list_mat
):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .mat array files
    and specify an annotation format
    but do not specify a labelset"""
    vak_df = vak.io.dataframe.from_files(
        data_dir=spect_dir_mat,
        labelset=None,
        annot_format="yarden",
        audio_format=None,
        spect_format="mat",
        annot_file=annot_file_yarden,
        spect_params=None,
    )

    assert returned_dataframe_matches_expected(
        vak_df,
        data_dir=spect_dir_mat,
        labelset=None,
        annot_format="yarden",
        audio_format=None,
        spect_format="mat",
        annot_file=annot_file_yarden,
        expected_spect_paths=spect_list_mat,
    )

    del default_spect_params['audio_path_key']  # 'audio_path' not strictly required
    assert spect_files_have_correct_keys(vak_df, default_spect_params)


def test_add_split_col(audio_dir_cbin,
                       default_spect_params,
                       labelset_notmat,
                       tmp_path):
    """test that ``add_split_col`` adds a 'split' column
    to a DataFrame, where all values in the Series are the
    specified split (a string)"""
    vak_df = vak.io.dataframe.from_files(
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

    vak_df = vak.io.dataframe.add_split_col(vak_df, split="train")
    assert "split" in vak_df.columns

    assert vak_df["split"].unique().item() == "train"
