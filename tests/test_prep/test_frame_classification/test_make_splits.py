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
                     input_type, tmp_path, specific_config_toml_path):
    toml_path = specific_config_toml_path(
        config_type,
        model_name,
        annot_format,
        audio_format,
        spect_format,
    )
    cfg = vak.config.parse.from_toml_path(toml_path)

    # ---- set up ----
    tmp_dataset_path = tmp_path / 'dataset_dir'
    tmp_dataset_path.mkdir()

    purpose = config_type

    def _prep_steps_up_to_make_splits():
        """copied almost verbatim from vak.prep.frame_classification
        this is all the stuff that happens before we call make_splits
        """
        if input_type == "spect":
            dataset_df = vak.prep.spectrogram_dataset.prep_spectrogram_dataset(
                labelset=cfg.prep.labelset,
                data_dir=cfg.prep.data_dir,
                annot_format=cfg.prep.annot_format,
                annot_file=cfg.prep.annot_file,
                audio_format=cfg.prep.audio_format,
                spect_format=cfg.prep.spect_format,
                spect_params=cfg.spect_params,
                spect_output_dir=tmp_dataset_path,
                audio_dask_bag_kwargs=cfg.prep.audio_dask_bag_kwargs,
            )
        elif input_type == "audio":
            dataset_df = vak.prep.audio_dataset.prep_audio_dataset(
                audio_format=cfg.prep.audio_format,
                data_dir=cfg.prep.data_dir,
                annot_format=cfg.prep.annot_format,
                labelset=cfg.prep.labelset,
            )

        if dataset_df.empty:
            raise ValueError(
                "Calling `vak.prep.spectrogram_dataset.prep_spectrogram_dataset` "
                "with arguments passed to `vak.core.prep` "
                "returned an empty dataframe.\n"
                "Please double-check arguments to `vak.core.prep` function."
            )

        # ---- (possibly) split into train / val / test sets ---------------------------------------------
        # catch case where user specified duration for just training set, raise a helpful error instead of failing silently
        if (purpose == "train" or purpose == "learncurve") and (
                (cfg.prep.train_dur is not None and cfg.prep.train_dur > 0)
                and (cfg.prep.val_dur is None or cfg.prep.val_dur == 0)
                and (cfg.prep.test_dur is None or cfg.prep.val_dur == 0)
        ):
            raise ValueError(
                "A duration specified for just training set, but prep function does not currently support creating a "
                "single split of a specified duration. Either remove the train_dur option from the prep section and "
                "rerun, in which case all data will be included in the training set, or specify values greater than "
                "zero for test_dur (and val_dur, if a validation set will be used)"
            )

        if all(
                [dur is None for dur in (cfg.prep.train_dur, cfg.prep.val_dur, cfg.prep.test_dur)]
        ) or purpose in (
                "eval",
                "predict",
        ):
            # then we're not going to split
            do_split = False
        else:
            if cfg.prep.val_dur is not None and cfg.prep.train_dur is None and cfg.prep.test_dur is None:
                raise ValueError(
                    "cannot specify only val_dur, unclear how to split dataset into training and test sets"
                )
            else:
                do_split = True

        if do_split:
            dataset_df = vak.prep.split.frame_classification_dataframe(
                dataset_df,
                tmp_dataset_path,
                labelset=cfg.prep.labelset,
                train_dur=cfg.prep.train_dur,
                val_dur=cfg.prep.val_dur,
                test_dur=cfg.prep.test_dur,
            )

        elif (
                do_split is False
        ):  # add a split column, but assign everything to the same 'split'
            # ideally we would just say split=purpose in call to add_split_col, but
            # we have to special case, because "eval" looks for a 'test' split (not an "eval" split)
            if purpose == "eval":
                split_name = (
                    "test"  # 'split_name' to avoid name clash with split package
                )
            elif purpose == "predict":
                split_name = "predict"

            dataset_df = vak.prep.dataset_df_helper.add_split_col(
                dataset_df, split=split_name
            )

        # ---- create and save labelmap ------------------------------------------------------------------------------------
        # we do this before creating array files since we need to load the labelmap to make frame label vectors
        if purpose != "predict":
            # TODO: add option to generate predict using existing dataset, so we can get labelmap from it
            map_unlabeled_segments = vak.prep.sequence_dataset.has_unlabeled_segments(
                dataset_df
            )
            labelmap = vak.common.labels.to_map(
                cfg.prep.labelset, map_unlabeled=map_unlabeled_segments
            )
            # save labelmap in case we need it later
            with (tmp_dataset_path / "labelmap.json").open("w") as fp:
                json.dump(labelmap, fp)
        else:
            labelmap = None

        return dataset_df, labelmap

    dataset_df, labelmap = _prep_steps_up_to_make_splits()

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

        assert vak.datasets.frame_classification.constants.FRAMES_PATH_COL_NAME in split_df.columns

        frames_paths = split_df[
            vak.datasets.frame_classification.constants.FRAMES_PATH_COL_NAME
        ].values

        sample_id_vecs, inds_in_sample_vecs = [], []
        # TODO: iterate through with frame paths + annot tuples at same time
        # TODO: so we can get frame times if needed
        # TODO: (if annot) an
        for sample_id, frames_path in enumerate(frames_paths):
            frames_file_that_should_exist = tmp_dataset_path / frames_path
            assert frames_file_that_should_exist.exists()

            # NOTE we load frames to confirm we can and also to make indexing vectors we use to test,
            # see next code block
            frames = vak.datasets.frame_classification.helper.load_frames(tmp_dataset_path / frames_path, input_type)
            assert isinstance(frames, np.ndarray)

            # make indexing vectors that we use to test
            n_frames = frames.shape[-1]
            sample_id_vecs.append(np.ones((n_frames,)).astype(np.int32) * sample_id)
            inds_in_sample_vecs.append(np.arange(n_frames))

        if purpose != "predict":
            assert vak.datasets.frame_classification.constants.FRAME_LABELS_NPY_PATH_COL_NAME in split_df.columns

            frame_labels_paths = split_df[
                vak.datasets.frame_classification.constants.FRAME_LABELS_NPY_PATH_COL_NAME
            ].values
            annots = vak.common.annotation.from_df(split_df)

            for frame_labels_path, annot in zip(frame_labels_paths, annots):
                frame_labels_file_that_should_exist = tmp_dataset_path / frame_labels_path
                assert frame_labels_file_that_should_exist.exists()

                lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
                expected_frame_labels = vak.transforms.frame_labels.from_segments(
                    lbls_int,
                    annot.seq.onsets_s,
                    annot.seq.offsets_s,
                    frame_times,
                    unlabeled_label=labelmap["unlabeled"],
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
            vak.datasets.frame_classification.constants.SAMPLE_IDS_ARRAY_FILENAME
        )
        assert sample_id_vec_path.exists()

        expected_sample_id_vec = np.concatenate(sample_id_vecs)
        sample_id_vec = np.load(sample_id_vec_path)
        assert np.array_equal(sample_id_vec, expected_sample_id_vec)

        inds_in_sample_vec_path = (
            split_subdir /
            vak.datasets.frame_classification.constants.INDS_IN_SAMPLE_ARRAY_FILENAME
        )
        assert inds_in_sample_vec_path.exists()

        expected_inds_in_sample_vec = np.concatenate(inds_in_sample_vecs)
        inds_in_sample_vec = np.load(inds_in_sample_vec_path)
        assert np.array_equal(inds_in_sample_vec, expected_inds_in_sample_vec)
